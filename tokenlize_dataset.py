from miditok import REMI
from pathlib import Path
import json

# 加载训练好的tokenizer
tokenizer_path = Path('tokenizer/tokenizer.json')  # tokenizer参数文件的路径
tokenizer = REMI(params=tokenizer_path)  # 使用参数路径初始化tokenizer

# 指定切分好的MIDI文件目录和tokenized数据的保存目录
splited_midi_dir = Path('dataset/splited')
tokenized_dataset_dir = Path('dataset_tokenlized')
tokenized_dataset_dir.mkdir(parents=True, exist_ok=True)

# 遍历每个切分好的MIDI文件
for midi_file in splited_midi_dir.glob('**/*.midi'):
    # Tokenize MIDI文件
    tokens = tokenizer(midi_file)
    
    # 准备存储tokenized数据的文件路径
    base_name = midi_file.stem.split('_bar_')[0]  # 获取原始MIDI文件的基本名称
    json_file_path = tokenized_dataset_dir / f'{base_name}.json'
    
    # 将tokens以行为单位写入JSON文件
    with json_file_path.open('a') as json_file:  # 使用追加模式打开文件
        json_file.write(json.dumps(tokens) + '\n')  # 将tokens转换为JSON字符串并追加新行

