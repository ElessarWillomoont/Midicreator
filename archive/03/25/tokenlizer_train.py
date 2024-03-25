from miditok import REMI, TokenizerConfig
from pathlib import Path

# 特殊token和其他参数
TOKENIZER_PARAMS = {
    "special_tokens": ["PAD" , "BOS", "EOS", "MASK"],  # 特殊token
    "num_velocities": 64,  # 力度数量
    "use_chords": True,  # 是否使用和弦
    "use_rests": True,  # 是否使用休止符
    "use_tempos": True,  # 是否使用速度
    "use_time_signatures": True,  # 是否使用拍号
    "use_programs": False,  # 是否使用乐器
}

# 创建TokenizerConfig实例
config = TokenizerConfig(**TOKENIZER_PARAMS)

# 使用配置创建tokenizer实例
tokenizer = REMI(config)
'''
# 指定MIDI文件和tokenizer保存的目录
midi_dir = Path('dataset')  # MIDI文件的根目录
tokenizer_dir = Path('tokenizer')  # tokenizer保存的目录

# 如果tokenizer保存目录不存在，则创建
tokenizer_dir.mkdir(parents=True, exist_ok=True)

# 列出所有MIDI文件的路径
midi_paths = list(midi_dir.glob("**/*.midi"))

# 确保midi_paths不为空
if not midi_paths:
    raise ValueError("No MIDI files found in the specified directory.")

# 使用BPE训练tokenizer
#tokenizer.learn_bpe(vocab_size=30000, files_paths=midi_paths)

# 保存训练好的tokenizer参数到文件
tokenizer.save_params(out_path=tokenizer_dir, filename='tokenizer.json')
'''
print(type(tokenizer.vocab))
print(tokenizer.special_tokens_ids)