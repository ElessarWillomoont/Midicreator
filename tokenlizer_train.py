from miditok import REMI
from pathlib import Path

# 创建tokenizer实例
tokenizer = REMI()

# 指定MIDI文件和tokenizer保存的目录
midi_dir = Path('maestro')  # MIDI文件的根目录
tokenizer_dir = Path('tokenizer')  # tokenizer保存的目录

# 如果tokenizer保存目录不存在，则创建
tokenizer_dir.mkdir(parents=True, exist_ok=True)

# 列出所有MIDI文件的路径
midi_paths = list(midi_dir.glob("**/*.mid"))

# 确保midi_paths不为空
if not midi_paths:
    raise ValueError("No MIDI files found in the specified directory.")

# 使用BPE训练tokenizer
tokenizer.learn_bpe(vocab_size=30000, files_paths=midi_paths)

# 保存训练好的tokenizer参数到文件
tokenizer.save_params(out_path=tokenizer_dir, filename='tokenizer.json')
