import os
from miditok import REMI, MIDITokenizer
from miditoolkit import MidiFile
from pathlib import Path

# 加载先前保存的tokenizer参数
tokenizer_path = Path('tokenizer/tokenizer.json')

# 初始化tokenizer，使用保存的参数
tokenizer = REMI(params=tokenizer_path)

# 设定数据集的路径
dataset_path = 'maestro'

# 计算数据集的总token数量，包括子目录中的文件
def calculate_total_tokens(dataset_path, tokenizer):
    total_tokens = 0
    for root, dirs, files in os.walk(dataset_path):
        for midi_file_name in files:
            if midi_file_name.endswith('.mid') or midi_file_name.endswith('.midi'):
                # 加载MIDI文件
                midi_path = os.path.join(root, midi_file_name)
                midi_data = MidiFile(midi_path)
                
                # 将MIDI数据转换为tokens
                # 注意：此处需要根据miditok的API调整
                tokens = tokenizer(midi_data)  # 确保这是正确的方法
                
                # 累加token数量
                total_tokens += len(tokens)  # tokens现在是一个直接的列表

    return total_tokens

# 估算token数量
total_tokens = calculate_total_tokens(dataset_path, tokenizer)
print(f'The dataset contains approximately {total_tokens} tokens.')