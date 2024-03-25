from pathlib import Path
from models.model_transformer import GPT2LikeTransformer  # 导入你自定义的模型类
from miditok import REMI  # 导入你训练tokenizer所使用的库

# 初始化并加载你的tokenizer
tokenizer_path = Path('tokenizer/tokenizer.json')  # tokenizer参数文件的路径
tokenizer = REMI(params=tokenizer_path)  # 使用参数路径初始化tokenizer

# 模型配置参数
n_layer=6
n_head=4
n_emb=32
context_length = 256
vocab_size = 30000  # 根据你的tokenizer实际情况调整
pad_token_id = 0  # 假设你的pad_token_id是0，根据实际情况调整

# 初始化你的模型
model = GPT2LikeTransformer(
    vocab_size=vocab_size, 
    n_layer=n_layer, 
    n_head=n_head, 
    n_emb=n_emb, 
    context_length=context_length, 
    pad_token_id=pad_token_id
)

# 计算模型的参数数量
model_size = sum(t.numel() for t in model.parameters())
print(f"Custom GPT-2-like model size: {model_size/1000**1:.1f}K parameters")
