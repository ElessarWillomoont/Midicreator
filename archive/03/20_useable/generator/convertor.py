from miditoolkit import MidiFile
from miditok import REMI
from pathlib import Path
from copy import deepcopy
import torch
import json

from miditok import REMI, TokenizerConfig
from symusic import Score

# Create a tokenizer config with your desired parameters
tokenizer_path = Path('tokenizer/tokenizer.json')

# Initialize tokenizer using the saved parameters
tokenizer = REMI(params=tokenizer_path)
input_ids_list = [228, 25, 231, 267, 755, 15, 3190, 15, 228, 25, 231, 267, 755, 15, 3190, 15,228, 25, 231, 267, 755, 15, 3190, 15,228, 25, 231, 267, 755, 15, 3190, 15]
tokens_no_bpe = tokenizer.decode_bpe(deepcopy(input_ids_list))
print(tokens_no_bpe)
# 假设您已经有了一系列token ids
#token_ids = [ ... ]  # 您的token ids列表

input_ids_list = [228, 25, 231, 267, 755, 15, 3190, 15, 228, 25, 231, 267, 755, 15, 3190, 15,228, 25, 231, 267, 755, 15, 3190, 15,228, 25, 231, 267, 755, 15, 3190, 15]
#tensor = torch.tensor(input_ids_list)

#midi_object = tokenizer.tokens_to_midi(tokens=input_ids_list, output_path="your_output_path.mid")
# 现在您可以保存midi对象为一个新的MIDI文件
#midi.write('path/to/your_new_midi.mid')