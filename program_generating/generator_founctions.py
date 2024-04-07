import sys
import os
script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(script_path))
sys.path.append(parent_directory)

from miditok import  TokSequence
import json
import torch
from shared.models import DecoderOnlyTransformer
import torch.nn.functional as F

CHECK_POINT = "shared/ckpt/ckpt_pretrained.pt"
MAX_LENGTH = 32
PAD_ID = 0
TEMPERATURE = 0.9  # 控制随机性，较低的值意味着较少的随机性
TOP_K = 50  # Top-K 抽样
TOP_P = 0.95  # Nucleus 抽样
TARGET_LENTH =128


def midi_file_to_token_ids(midi_file,tokenizer):
    """ Convert a MIDI file to a token sequence representation, now directly to ids"""
    tokens = tokenizer(midi_file)
    tokens = tokens[0]
    return tokens.ids
def token_ids_to_midis(token_ids,tokenizer,midi_file_output,vocabulary_json_file):
    tokens = tokens_to_vocabulary(token_ids, vocabulary_json_file)
    tokens = [TokSequence(tokens)]
    midi = tokenizer._tokens_to_midi(tokens)
    midi.dump_midi(midi_file_output)
    print(f"MIDI file saved as {midi_file_output}")
    return midi
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
def adjust_logits_for_structure(class_ranges,sequence_structure,output_logits, structure_index, insert_oth=False):
    """
    根据当前的结构索引和是否插入oth类别来调整logits。
    """
    # 初始化所有logits为非常小的值
    output_logits += -float('Inf')
    
    if insert_oth:
        # 激活oth范围内的logits
        start, end = class_ranges['oth']
        output_logits[:, start:end] = 0
    else:
        # 根据当前结构索引激活相应范围内的logits
        category = sequence_structure[structure_index]
        start, end = class_ranges[category]
        output_logits[:, start:end] = 0
    return output_logits

def process_midi(input_midi_file, output_midi_file,tokenizer,vocabulary_json_file):
    # Create a tokenizer config with your desired parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperature = TEMPERATURE
    top_k = TOP_K
    top_p = TOP_P
    # Convert input MIDI to tokens
    input_ids = midi_file_to_token_ids(input_midi_file,tokenizer)
    print("Input MIDI converted to tokens.")
    # 定义输出结构和类别范围
    sequence_structure = ['pos', 'tem','pitch', 'vol', 'dur']
    class_ranges = {
        'pitch': (5, 93),
        'vol': (93, 157),
        'dur': (157, 221),
        'pos': (221, 317),
        'oth': (317, 392),
        'tem': (424, 455)
    }
    structure_index = 0  # 当前结构索引
    prob_insert_oth = 0.1  # 插入oth类别元素的概率
    model = DecoderOnlyTransformer(vocab_size=465, decoder_layer=6, n_head=4, n_emb=768, context_length=MAX_LENGTH, pad_token_id=PAD_ID)
    checkpoint = torch.load(CHECK_POINT,map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.eval()
    input_ids_list = input_ids
    # Start with the initial input
    current_input = input_ids_list

    # Placeholder for generated tokens
    generated_tokens = []

    # Loop until 128 tokens are generated
    while len(generated_tokens) < TARGET_LENTH:
        # Ensure the input is always 32 tokens
        if len(current_input) > MAX_LENGTH:
            current_input = current_input[-MAX_LENGTH:]  # Keep only the last MAX_LENGTH tokens

        # Convert to tensor
        input_tensor = torch.tensor([current_input], dtype=torch.long)
        
        # Create an attention mask for the padded input tensor
        attention_mask = input_tensor != 0
        
        # Model inference
        with torch.no_grad():
            output = model(input_tensor, attention_mask=attention_mask)

        output_logits = output.logits[:, -1, :] / temperature  # Apply temperature scaling to the last time step logits

        # Decide whether to insert 'oth'
        insert_oth = torch.rand(1).item() < prob_insert_oth
        output_logits = adjust_logits_for_structure(class_ranges,sequence_structure,output_logits, structure_index, insert_oth)

        # Apply Top-K and Top-P sampling to the adjusted logits
        # 对logits应用Top-K 抽样
        if top_k > 0:
            indices_to_remove = output_logits < torch.topk(output_logits, top_k)[0][..., -1, None]
            output_logits[indices_to_remove] = -float('Inf')

        # 对logits应用Top-P (Nucleus) 抽样
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(output_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 移除累积概率低于top_p的tokens，使得剩下的tokens的累积概率刚好大于top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # 保留第一个超过top_p的tokens，因为它是使得累积概率超过top_p的最小tokens集合
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            output_logits[:, indices_to_remove] = -float('Inf')

        probabilities = F.softmax(output_logits, dim=-1)
        last_predicted_token = torch.multinomial(probabilities, 1).item()
        
        # Append the predicted token to the list of generated tokens
        generated_tokens.append(last_predicted_token)
        
        # Update the input with the newly generated token
        current_input.append(last_predicted_token)

        if not insert_oth:
            # Move to the next element in the structure
            structure_index = (structure_index + 1) % len(sequence_structure)

    # Output the generated tokens
    print("Generated tokens:", generated_tokens)
    
    token_ids_to_midis(generated_tokens,tokenizer,output_midi_file,vocabulary_json_file)
    print(f"MIDI file saved as {output_midi_file}")
