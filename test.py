from qwen2 import Qwen2ForCausalLM
from moe.model import MoEModel
from moe.config import MoEConfig

llm_model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B")
moe_config = MoEConfig(
    num_experts=4,          # 每层的专家数量
    topk=2,                 # 每次选择的专家数量
    aux_loss_coef=0.01,     # 辅助损失系数（负载平衡）
    lpr_loss_coef=None,     # 路由稀疏性损失系数
    init_moe_weights=True,  # 是否初始化专家权重
    layers_to_transform=list(range(len(llm_model.model.layers))),  # 替换的目标层（按层编号）
)

# 加载预训练的 LLM 模型

# 初始化 MoE PEFT
adapter_name = "moe_adapter"
moe_model = MoEModel(
    model=llm_model, 
    config=moe_config, 
    adapter_name=adapter_name
)

print(moe_model)
# 打印哪些参数是可训练的
for n, p in moe_model.named_parameters():
    print(n, p.requires_grad)