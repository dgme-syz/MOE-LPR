import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from peft.tuners.tuners_utils import BaseTunerLayer

class MoELayer(BaseTunerLayer):
    # Used to replace ffn layer in moe layer
    adapter_layer_names = (
        "moe_expert", "moe_router"
    )
    
    def __init__(self, ffn_layer):
        self.ffn_layer = ffn_layer
        self.moe_router = nn.ModuleDict({})
        self.moe_expert = nn.ModuleDict({})
    
        # from ffn layer get input_dim, qwen2 use gate_proj to get input_dim 
        if hasattr(ffn_layer, "gate_proj"):
            self.input_dim = ffn_layer.gate_proj.in_features
        else:
            raise ValueError("ffn layer should have gate_proj attribute, but not found")
        
    def update_layer(self, ffn_layer, adapter_name, num_experts, init_moe_weights=True):
        self.moe_router[adapter_name] = nn.Linear(self.input_dim, num_experts, bias=False)
        # just copy ffn layer for each expert, so we can keep old layer's parameters
        self.reset_moe_parameters(adapter_name) # init router weight
        self.moe_expert[adapter_name] = nn.ModuleList([copy.deepcopy(ffn_layer) for _ in range(num_experts - 1)])
        
        if init_moe_weights:
            self.reset_moe_parameters(adapter_name)
        
        self.set_adapter(self.active_adapters)
        
    def reset_moe_parameters(self, adapter_name):
        if adapter_name in self.moe_router.keys():
            nn.init.xavier_normal_(self.moe_router[adapter_name].weight)
            
class MoE(nn.Module, MoELayer):
    def __init__(
        self, 
        ffn_layer,
        adapter_name,
        num_experts,
        init_moe_weights,
        topk,
        aux_loss_coef,
        lpr_loss_coef,
        **kwargs
    ):
        super(MoE, self).__init__()
        MoELayer.__init__(self, ffn_layer)
        self.update_layer(ffn_layer, adapter_name, num_experts)
        
        self.topk = topk
        self.aux_loss_coef = aux_loss_coef
        self.lpr_loss_coef = lpr_loss_coef
        
    def forward(self, x):
        dtype = x.dtype
        router = self.moe_router[self.active_adapters[0]]
        result, router_logits = self.topk_router(x, router, self.active_adapters[0])
        result = result.to(dtype)
        return result, router_logits

    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L293
    def topk_route(self, x, router, adapter=None):
        batch_size, seq_len, hidden_dim = x.shape
        x = x.view(-1, hidden_dim)
        router_logits = router(x) # (batch_size * seq_len, num_experts)
        
        router_weights = F.softmax(router_logits, dim=-1)
        router_weights, selected_experts = torch.topk(router_weights, self.topk, dim=-1) #(batch_size * seq_len, topk)
        
        if self.topk != 1:
            router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        
        router_weights = router_weights.to(x.dtype)
        
        final_hidden_state = torch.zeros(
            (batch_size * seq_len, hidden_dim), dtype=x.dtype, device=x.device
        )
        num_experts = router_logits.shape[-1] + 1
        expert_mask = F.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0) #(num_experts, num_experts, batch_size * seq_len)
        experts = [self.ffn_layer] + self.moe_expert[adapter]
        
        for expert_idx in range(num_experts):
            expert_layer = experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            current_state = x[None, top_x].reshape(-1, hidden_dim)
            current_hidden_state = expert_layer(current_state) * router_weights[top_x, idx, None]
            final_hidden_state.index_add_(0, top_x, current_hidden_state.to(x.dtype))
        final_hidden_state = final_hidden_state.view(batch_size, seq_len, hidden_dim)
        return final_hidden_state, router_logits