from miditoolkit import MidiFile
from miditok import REMI, TokSequence
from pathlib import Path
from copy import deepcopy
import json
from miditok import REMI, TokenizerConfig
from symusic import Score

def save_vocabulary_to_json(vocabulary, file_name='vocabulary.json'):
    """
    将词汇表保存到JSON文件中。

    参数:
    - vocabulary: 一个字典，包含要保存的词汇表。
    - file_name: 字符串，输出文件的名称，默认为'vocabulary.json'。
    """
    with open(file_name, 'w') as json_file:
        json.dump(vocabulary, json_file, indent=4)

def tokens_to_vocabulary(token_ids, json_file_path):
    """
    根据token ids和词汇表字典输出对应的词汇列表。

    参数:
    - token_ids: 一个包含多个int类型的token ids的列表。
    - json_file_path: 包含词汇表字典的json文件路径。
    
    返回:
    - 一个包含对应词汇的列表。
    """
    # 从JSON文件加载词汇表字典
    with open(json_file_path, 'r') as json_file:
        vocabulary = json.load(json_file)
    
    # 反转词汇表字典，以便通过id查找词汇
    id_to_vocab = {v: k for k, v in vocabulary.items()}
    
    # 根据token ids查找对应的词汇
    vocab_list = [id_to_vocab[token_id] for token_id in token_ids if token_id in id_to_vocab]
    
    return vocab_list

# Create a tokenizer config with your desired parameters
tokenizer_path = Path('tokenizer/tokenizer.json')

# Initialize tokenizer using the saved parameters
tokenizer = REMI(params=tokenizer_path)

#print(tokenizer.vocab)

# 假设这是您的词汇表
vocabulary = tokenizer.vocab
json_file = 'vocabulary.json'
ids = [61, 157, 126, 392, 48, 126, 157, 157, 392, 126, 126, 157, 157, 392, 157, 157, 392, 157, 31, 131, 31, 31, 392, 31, 43, 392, 31, 31, 392, 133, 31, 157, 31, 131, 157, 133, 133, 129, 36]


tokens = tokens_to_vocabulary(ids, json_file)
tokens = TokSequence([[tokens]])
midi = tokenizer._tokens_to_midi(tokens)
midi.dump_midi('output.midi')
print(midi)



# input('say sth')

# midi_seq = [
# 28, 114, 160, 16, 112, 164, 246, 18, 114, 158, 30, 118, 158, 399, 392, 225, 20, 112, 174, 44, 119 
# ]
# print(midi_seq)
# midi_seq = [tokenizer._ids_to_tokens([tok])[0] for tok in midi_seq]
# print(midi_seq)

# #midi = tokenizer.tokens_to_midi(midi_seq)