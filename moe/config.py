from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class MoEConfig(PeftConfig):
    init_moe_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the Moe layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    num_experts: int = field(
        default=2,
        metadata={
            "help": (
                "The total number of experts for moe fine-tuning. If set to N, then N-1 new experts are added."
            ),
        },
    )
    topk: int = field(
        default=1,
        metadata={
            "help": (
                "How much experts are selected for each token."
            ),
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": "Upcycling to MoE layer for which layers."
        },
    )
    save_all_params: bool = field(
        default=False,
        metadata={
            "help": (
                "Updates and save all the parameters of MoE."
            ),
        },
    )
    aux_loss_coef: float = field(
        default=None,
        metadata={
            "help": (
                "The weight of the load balancing loss. Only will be used if set."
            ),
        },
    )
    lpr_loss_coef: float = field(
        default=None,
        metadata={
            "help": (
                "The weight of the lpr loss. Only will be used if set."
            ),
        },
    )

    def __post_init__(self):
        self.peft_type = "MoE"
        if self.lpr_loss_coef is not None and self.aux_loss_coef is not None:
            raise NotImplementedError