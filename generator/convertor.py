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
print(tokenizer.vocab)
input_ids_list = 'Tempo_53.55'
midi = converted_back_midi = tokenizer.tokens_to_midi([input_ids_list])
# 现在您可以保存midi对象为一个新的MIDI文件
#midi.write('path/to/your_new_midi.mid')