import sys
import os
import json
import torch
import torch.nn.functional as F
from miditok import TokSequence
import shared.config as configue
from shared.models import DecoderOnlyTransformer

# Adjust the system path to import the configuration file from a parent directory
script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(script_path))
sys.path.append(parent_directory)

# Load configurations from the external configuration file
CHECK_POINT = configue.CHECK_POINT_USE
MAX_LENGTH = configue.MAX_LENGTH
PAD_ID = configue.PAD_ID
TEMPERATURE = configue.TEMPERATURE  # Controls randomness, lower values imply less randomness
TOP_K = configue.TOP_K  # Top-K sampling
TOP_P = configue.TOP_P  # Nucleus sampling
TARGET_LENGTH = configue.TARGET_LENTH


def midi_file_to_token_ids(midi_file, tokenizer):
    """Converts a MIDI file to a token sequence representation, now directly to ids."""
    tokens = tokenizer(midi_file)
    tokens = tokens[0]
    return tokens.ids

def token_ids_to_midis(token_ids, tokenizer, midi_file_output, vocabulary_json_file):
    """Converts token IDs back to MIDI, saves the MIDI file."""
    tokens = tokens_to_vocabulary(token_ids, vocabulary_json_file)
    tokens = [TokSequence(tokens)]
    midi = tokenizer._tokens_to_midi(tokens)
    midi.dump_midi(midi_file_output)
    print(f"MIDI file saved as {midi_file_output}")
    return midi

def save_vocabulary_to_json(vocabulary, file_name='vocabulary.json'):
    """Saves the vocabulary to a JSON file."""
    with open(file_name, 'w') as json_file:
        json.dump(vocabulary, json_file, indent=4)

def tokens_to_vocabulary(token_ids, json_file_path):
    """Converts token ids to their corresponding vocabulary words based on the vocabulary dictionary."""
    # Load the vocabulary dictionary from a JSON file
    with open(json_file_path, 'r') as json_file:
        vocabulary = json.load(json_file)
    
    # Reverse the vocabulary dictionary to map ids back to vocabulary
    id_to_vocab = {v: k for k, v in vocabulary.items()}
    
    # Map each token id to its corresponding vocabulary word
    vocab_list = [id_to_vocab[token_id] for token_id in token_ids if token_id in id_to_vocab]
    
    return vocab_list

def adjust_logits_for_structure(class_ranges, sequence_structure, output_logits, structure_index, insert_oth=False):
    """Adjusts logits based on the current structure index and whether to insert 'oth' category."""
    # Initialize all logits to a very small value
    output_logits += -float('Inf')
    
    if insert_oth:
        # Activate logits within the 'oth' range
        start, end = class_ranges['oth']
        output_logits[:, start:end] = 0
    else:
        # Activate logits within the range of the current structure index
        category = sequence_structure[structure_index]
        start, end = class_ranges[category]
        output_logits[:, start:end] = 0
    return output_logits

def process_midi(input_midi_file, output_midi_file, tokenizer, vocabulary_json_file):
    """Processes the MIDI file by converting it to tokens, generating new tokens, and converting back to MIDI."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert input MIDI to tokens
    input_ids = midi_file_to_token_ids(input_midi_file, tokenizer)
    print("Input MIDI converted to tokens.")
    
    # Define the output structure and class ranges
    sequence_structure = ['pos', 'tem', 'pitch', 'vol', 'dur']
    class_ranges = {
        'pitch': (5, 93),
        'vol': (93, 157),
        'dur': (157, 221),
        'pos': (221, 317),
        'oth': (317, 392),
        'tem': (424, 455)
    }
    structure_index = 0  # Current structure index
    prob_insert_oth = 0.1  # Probability of inserting an 'oth' category element
    
    # Load the model and checkpoint
    model = DecoderOnlyTransformer(vocab_size=configue.vocab_size, decoder_layer=configue.decoder_layer, n_head=configue.n_head, n_emb=configue.n_emb, context_length=MAX_LENGTH, pad_token_id=PAD_ID)
    checkpoint = torch.load(CHECK_POINT, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.eval()
    
    input_ids_list = input_ids
    current_input = input_ids_list  # Start with the initial input
    generated_tokens = []  # Placeholder for generated tokens

    # Generate tokens until the target length is reached
    while len(generated_tokens) < TARGET_LENGTH:
        # Ensure the input is always of MAX_LENGTH
        if len(current_input) > MAX_LENGTH:
            current_input = current_input[-MAX_LENGTH:]  # Keep only the last MAX_LENGTH tokens

        # Model inference
        input_tensor = torch.tensor([current_input], dtype=torch.long)
        attention_mask = input_tensor != 0
        with torch.no_grad():
            output = model(input_tensor, attention_mask=attention_mask)

        output_logits = output.logits[:, -1, :] / TEMPERATURE  # Apply temperature scaling

        # Decide whether to insert 'oth'
        insert_oth = torch.rand(1).item() < prob_insert_oth
        output_logits = adjust_logits_for_structure(class_ranges, sequence_structure, output_logits, structure_index, insert_oth)

        # Apply Top-K and Top-P sampling
        output_logits = apply_sampling(output_logits, TOP_K, TOP_P)

        probabilities = F.softmax(output_logits, dim=-1)
        last_predicted_token = torch.multinomial(probabilities, 1).item()
        generated_tokens.append(last_predicted_token)
        current_input.append(last_predicted_token)

        if not insert_oth:
            structure_index = (structure_index + 1) % len(sequence_structure)  # Move to the next element in the structure

    print("Generated tokens:", generated_tokens)
    
    token_ids_to_midis(generated_tokens, tokenizer, output_midi_file, vocabulary_json_file)
    print(f"MIDI file saved as {output_midi_file}")

def apply_sampling(output_logits, top_k, top_p):
    """Applies Top-K and Top-P sampling to logits."""
    # Apply Top-K sampling
    if top_k > 0:
        indices_to_remove = output_logits < torch.topk(output_logits, top_k)[0][..., -1, None]
        output_logits[indices_to_remove] = -float('Inf')

    # Apply Top-P (Nucleus) sampling
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(output_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        output_logits[:, indices_to_remove] = -float('Inf')
    
    return output_logits
