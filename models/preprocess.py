# preprocess_data.py
import json
import os

MAX_LENGTH = 256  # 设定全局变量MAX_LENGTH

def preprocess_data(input_dir, output_file):
    data_pairs = []
    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'): 
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as f:
                    for line in f:
                        music_data = json.loads(line)
                        ids = music_data['ids'][0]  # 假设我们只处理第一个元素
                        input_ids = [1] + ids + [2]  # 添加BOS和EOS
                        target_ids = [1] + ids + [2]  # 添加BOS和EOS
                        # 截断或填充input_ids和target_ids到MAX_LENGTH
                        input_ids = (input_ids + [0] * MAX_LENGTH)[:MAX_LENGTH]
                        target_ids = (target_ids + [0] * MAX_LENGTH)[:MAX_LENGTH]
                        data_pairs.append((input_ids, target_ids))
    
    with open(output_file, 'w') as f:
        json.dump(data_pairs, f)

preprocess_data('dataset/dataset_json/train_set', 'dataset/train_data.json')
preprocess_data('dataset/dataset_json/validation_set', 'dataset/validation_data.json')
