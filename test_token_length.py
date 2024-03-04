from miditok import REMI
from pathlib import Path
import numpy as np

# 加载训练好的tokenizer
tokenizer_path = Path('tokenizer/tokenizer.json')  # tokenizer参数文件的路径
tokenizer = REMI(params=tokenizer_path)  # 使用参数路径初始化tokenizer

# 指定MIDI文件的目录
midi_dir = Path('dataset')

# 列出所有MIDI文件的路径
midi_paths = list(midi_dir.glob("**/*.midi"))

# 确保midi_paths不为空
if not midi_paths:
    raise ValueError("No MIDI files found in the specified directory.")

# 用于存储所有文件的token size
token_sizes = []

# 遍历所有MIDI文件
for midi_path in midi_paths:
    # 使用tokenizer对MIDI文件进行tokenize
    tokens = tokenizer(midi_path)
    tok_sequence = tokens[0]
    actual_tokens = tok_sequence.tokens
    ##print(f"实际的token数量是: {len(actual_tokens)}")
    ##input('stop here')
    # 记录当前文件的token size
    token_sizes.append(len(actual_tokens))

# 计算95%位的token size
percentile_95 = np.percentile(token_sizes, 5)

print(f"95%位的token size是: {percentile_95}")
