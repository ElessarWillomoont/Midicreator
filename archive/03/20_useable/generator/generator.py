import torch
from models import DecoderOnlyTransformer
import torch.nn.functional as F

CHECK_POINT = "model_output/checkpoints/ckpt_180000.pt"

# Load the model and checkpoint
model = DecoderOnlyTransformer(vocab_size=30000, decoder_layer=12, n_head=2, n_emb=768, context_length=8, pad_token_id=0)
checkpoint = torch.load(CHECK_POINT)
model.load_state_dict(checkpoint)
model.eval()

# Input processing
input_data = input("input ids :")
input_ids_list = [int(x) for x in input_data.split(",")]

# Start with the initial input
current_input = input_ids_list

# Placeholder for generated tokens
generated_tokens = []

# Loop until 128 tokens are generated
while len(generated_tokens) < 128:
    # Ensure the input is always 8 tokens
    if len(current_input) > 8:
        current_input = current_input[-8:]  # Keep only the last 8 tokens

    # Convert to tensor
    input_tensor = torch.tensor([current_input], dtype=torch.long)
    
    # Create an attention mask for the padded input tensor
    attention_mask = input_tensor != 0
    
    # Model inference
    with torch.no_grad():
        output = model(input_tensor, attention_mask=attention_mask)

    # Access the logits and get the last token predicted
    output_logits = output.logits
    _, predicted_token_ids = torch.max(output_logits, dim=2)
    last_predicted_token = predicted_token_ids[0, -1].item()  # Get the last token
    
    # Append the predicted token to the list of generated tokens
    generated_tokens.append(last_predicted_token)
    
    # Update the input with the newly generated token
    current_input.append(last_predicted_token)

# Output the generated tokens
print("Generated tokens:", generated_tokens)
