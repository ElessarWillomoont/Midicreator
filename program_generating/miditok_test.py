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
ids = [328, 284, 452, 326, 86, 123, 212, 249, 344, 358, 437, 65, 98, 167, 366, 236, 446, 10, 139, 168, 306, 447, 60, 329, 102, 190, 292, 427, 66, 132, 189, 248, 442, 81, 95, 189, 259, 436, 50, 128, 210, 247, 425, 319, 45, 112, 210, 273, 439, 37, 116, 190, 307, 444, 375, 24, 386, 96, 169, 274, 427, 13, 154, 193, 307, 452, 21, 154, 217, 313, 440, 73, 107, 215, 281, 432, 91, 126, 203, 299, 330, 430, 82, 112, 203, 281, 451, 327, 11, 115, 199, 297, 390, 445, 56, 151, 371, 191, 225, 451, 52, 107, 199, 316, 440, 60, 154, 164, 223, 440, 15, 107, 196, 292, 453, 373, 361, 64, 97, 167, 317, 307, 433, 75, 129, 167, 299, 441]


tokens = tokens_to_vocabulary(ids, json_file)
print(tokens)
input("say sth")
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