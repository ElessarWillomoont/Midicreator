import torch
from models import DecoderOnlyTransformer
import torch.nn.functional as F

CHECK_POINT = "model_output/checkpoints/ckpt_48000.pt"
MAX_LENGTH = 32
PAD_ID = 0

# Load the model and checkpoint
model = DecoderOnlyTransformer(vocab_size=465, decoder_layer=6, n_head=4, n_emb=768, context_length=MAX_LENGTH, pad_token_id=PAD_ID)
checkpoint = torch.load(CHECK_POINT)
model.load_state_dict(checkpoint)
model.eval()

# Input processing
input_data = input("input ids :")
input_ids_list = [int(x) for x in input_data.split(",")]
input_tensor = torch.tensor([input_ids_list], dtype=torch.long)
padding_needed = max(0, 8 - input_tensor.shape[1])
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
