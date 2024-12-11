from .config import MoeConfig
from .layer import MoeLayer, MLP
from .model import MoeModel

import sys
from peft.utils.save_and_load import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

def support_moe_save(func):
    def wrapper(model, state_dict=None, adapter_name="default", unwrap_compiled=False, save_embedding_layers="auto"):
        config = model.peft_config[adapter_name]
        if state_dict is None:
            state_dict = model.state_dict()
        if config.peft_type == PeftType.MoE:
            if config.save_all_params:
                return state_dict
            else:
                to_return = {k: state_dict[k] for k in state_dict if "moe_" in k}
                to_return = {k: v for k, v in to_return.items() if (("moe_" in k and adapter_name in k))}
        else:
            to_return = func(model, state_dict, adapter_name, unwrap_compiled, save_embedding_layers)
        return to_return
    return wrapper

def support_moe_load(func):
    def wrapper(model, peft_model_state_dict, adapter_name="default", ignore_mismatched_sizes: bool = False):
        config = model.peft_config[adapter_name]
        if config.peft_type == PeftType.MoE:
            model.load_state_dict(peft_model_state_dict, strict=False)
            return model
        else:
            return func(model, peft_model_state_dict, adapter_name, ignore_mismatched_sizes)
    return wrapper

# Add support for MoE
save = support_moe_save(get_peft_model_state_dict)
load = support_moe_load(set_peft_model_state_dict)
sys.modules["peft.peft_model"].get_peft_model_state_dict = save
sys.modules["peft.utils"].set_peft_model_state_dict = load

from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft.utils.peft_types import PeftType

# make peft support MoE
print("moe/__init__.py executed")
PeftType.MoE = "MoE"
PEFT_TYPE_TO_MODEL_MAPPING["MoE"] = MoeModel
PEFT_TYPE_TO_CONFIG_MAPPING["MoE"] = MoeConfig
PEFT_TYPE_TO_TUNER_MAPPING["MoE"] = MoeModel


# update get_peft_model_state_dict


__all__ = ["MoeConfig", "MoeLayer", "MLP", "MoeModel"]

