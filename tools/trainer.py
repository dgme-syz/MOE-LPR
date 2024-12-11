from typing import Dict
import torch
from transformers import Trainer
from transformers.utils import is_torch_tpu_available

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

# https://github.com/zjwang21/MoE-LPR/blob/master/transformers/src/transformers/models/qwen2/modeling_qwen2.py#L1357

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_loss_logged = False
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)
            self.main_loss_logged = True # add

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        balance_loss = outputs.balance_loss
        lpr_loss = outputs.lpr_loss

        if balance_loss != None:
            nlayers = len(outputs.router_logits)
            prefix = "load_balance_"
            with torch.no_grad():
                mask = inputs['attention_mask'].reshape(-1) # n
                mask = mask.unsqueeze(0).expand(nlayers, mask.size(0)).bool() # nlayers x n
                router_logits = outputs.router_logits
                router_logits = torch.stack(router_logits) # nlayers x n x e
                probs = torch.nn.functional.softmax(router_logits, dim=-1) # nlayers x n x e
                probs = torch.mean(probs[mask], dim=0).detach().cpu() # e

            logs: Dict[str, float] = {}
            if self.main_loss_logged:
                logs[f"{prefix}_loss"] = balance_loss.item()
                logs["scores_per_expert"] = " ".join([str(round(k, 2)) for k in probs.tolist()])
                self.log(logs)

        if lpr_loss != None:
            prefix = "lpr_"
            lang_mask = inputs['langs']
            with torch.no_grad():
                router_logits = outputs.router_logits
                router_logits = torch.stack(router_logits) # nlayers x n x e
                probs = torch.nn.functional.softmax(router_logits, dim=-1)
                mask = lang_mask.reshape(-1).bool().expand(probs.size()[:2])
                probs = probs[mask].to(torch.float).reshape(mask.size(0), -1, probs.size(-1)) # nlayers x n x e
                probs = probs[:, :, 0] # nlayers x n
                score_expert0 = torch.mean(probs, dim=-1).detach().cpu()

            logs: Dict[str, float] = {}
            if self.main_loss_logged:
                logs[f"{prefix}_loss"] = lpr_loss.item()
                logs["old_lang_expert0_score"] = " ".join([str(round(k, 2)) for k in score_expert0.tolist()])
                self.log(logs)

        if self.main_loss_logged: self.main_loss_logged = False
        return loss