import os
import json
from tqdm import tqdm
from miditok import REMI, TokenizerConfig  # 假设你使用的是REMI tokenizer
from pathlib import Path

tokenizer_path = Path('tokenizer/tokenizer.json')  # tokenizer参数文件的路径
tokenizer = REMI(params=tokenizer_path)  # 使用参数路径初始化tokenizer

def tokenize_midi_files_and_combine(splited_dir, dataset_dir):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    source_files = {}  # 用于存储源文件名和对应的切分文件路径

    # 遍历切分的MIDI文件，按源文件名组织
    for file in tqdm(os.listdir(splited_dir), desc="Organizing MIDI files"):
        if file.lower().endswith('.midi'):
            source_name = file.split('_bar_')[0]  # 假设文件名格式为"sourceName_bar_i.midi"
            if source_name not in source_files:
                source_files[source_name] = []
            source_files[source_name].append(os.path.join(splited_dir, file))

    # 对每个源文件的切分进行tokenize，并合并结果
    for source_name, files in tqdm(source_files.items(), desc="Tokenizing MIDI files"):
        # 使用tokenizer的tokenize_midi_dataset方法
        # 注意：你需要根据实际方法的参数和返回值调整下面的代码
        tokens_list = tokenizer.tokenize_midi_dataset(files, dataset_dir)  # 假设这个方法直接返回token序列的列表

        # 将token序列保存到JSON文件中
        combined_file_path = os.path.join(dataset_dir, f"{source_name}.json")
        with open(combined_file_path, 'w') as f:
            for tokens in tokens_list:
                f.write(json.dumps(tokens) + '\n')  # 将每个token序列作为单独的行写入

splited_dir = 'dataset/splited'  # 切分后的MIDI文件目录
dataset_dir = 'dataset/dataset_json'  # 最终的JSON文件目录

# 对切分的MIDI文件进行tokenize并按源文件组织
tokenize_midi_files_and_combine(splited_dir, dataset_dir)