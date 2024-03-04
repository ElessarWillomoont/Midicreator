import os
from miditok import REMI
from miditoolkit import MidiFile  # 确保从 miditoolkit 导入 MidiFile

# 初始化 tokenizer
tokenizer = REMI()

# 设定数据集的路径
dataset_path = 'maestro'

# 计算数据集的总 token 数量，包括子目录中的文件
def calculate_total_tokens(dataset_path, tokenizer):
    total_tokens = 0
    for root, dirs, files in os.walk(dataset_path):
        for midi_file_name in files:
            if midi_file_name.endswith('.mid') or midi_file_name.endswith('.midi'):
                # 加载 MIDI 文件
                midi_path = os.path.join(root, midi_file_name)
                midi_data = MidiFile(midi_path)  # 使用 miditoolkit 的 MidiFile 加载 MIDI
                
                # 将 MIDI 数据转换为 tokens
                tokens = tokenizer(midi_data)  # 曾经使用 encode 方法
                
                # 累加 token 数量
                total_tokens += len(tokens)  # tokens 现在是一个直接的列表

    return total_tokens

# 估算 token 数量
total_tokens = calculate_total_tokens(dataset_path, tokenizer)
print(f'The dataset contains approximately {total_tokens} tokens.')
