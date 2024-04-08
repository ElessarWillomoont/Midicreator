import torch
from shared.models import DecoderOnlyTransformer
import torch.nn.functional as F
import sys
import os
script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(script_path))
sys.path.append(parent_directory)
import shared.config as configue
CHECK_POINT = configue.CHECK_POINT_CONTINUE
MAX_LENGTH = configue.MAX_LENGTH
PAD_ID = configue.PAD_ID

# Load the model and checkpoint
model = DecoderOnlyTransformer(vocab_size=configue.vocab_size, decoder_layer=configue.decoder_layer, n_head=configue.n_head, n_emb=configue.n_emb, context_length=MAX_LENGTH, pad_token_id=PAD_ID)
checkpoint = torch.load(CHECK_POINT)
model.load_state_dict(checkpoint)
model.eval()

# Input processing
input_data = input("input ids :")
input_ids_list = [int(x) for x in input_data.split(",")]
input_tensor = torch.tensor([input_ids_list], dtype=torch.long)
padding_needed = max(0, 32 - input_tensor.shape[1])
input_tensor_padded = F.pad(input_tensor, (0, padding_needed), 'constant', 0)

# Create an attention mask for the padded input tensor
attention_mask = input_tensor_padded != 0

# Model inference with attention mask
with torch.no_grad():
    output = model(input_tensor_padded, attention_mask=attention_mask)

# Access the logits from the model's output
output_logits = output.logits

# Apply `torch.max` to the logits to get the predicted token IDs
_, predicted_token_ids = torch.max(output_logits, dim=2)

# Output the predicted token IDs
print("Token IDs:", predicted_token_ids.squeeze().tolist())
