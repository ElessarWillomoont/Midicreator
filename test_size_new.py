import os
from miditok import REMI
from miditok import MidiFile

# 初始化tokenizer
tokenizer = REMI()

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

                # miditok期望的属性名是 'ticks_per_quarter'，但miditoolkit使用的是 'ticks_per_beat'
                # 我们在这里创建一个别名，使之兼容
                if not hasattr(midi_data, 'ticks_per_quarter'):
                    midi_data.ticks_per_quarter = midi_data.ticks_per_beat
                
                # 直接将MIDI数据转换为tokens，跳过预处理步骤
                tokens = tokenizer.midi_to_tokens(midi_data)
                
                # 累加token数量
                total_tokens += len(tokens[0])  # tokens是一个列表的列表，我们只关心第一个元素（tracks）

    return total_tokens

# 估算token数量
total_tokens = calculate_total_tokens(dataset_path, tokenizer)
print(f'The dataset contains approximately {total_tokens} tokens.')

