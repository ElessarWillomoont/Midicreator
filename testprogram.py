from miditok import REMI
from pathlib import Path

# 加载训练好的tokenizer
tokenizer_path = Path('tokenizer/tokenizer.json')  # tokenizer参数文件的路径
tokenizer = REMI(params=tokenizer_path)  # 使用参数路径初始化tokenizer

# 打印PAD令牌的索引
print(tokenizer.special_tokens_ids)
