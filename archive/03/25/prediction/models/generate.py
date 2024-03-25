import torch
from model_test import TransformerModel
import torch.nn.functional as F

CHECK_POINT = "model_output/checkpoints/step_160000.pt"

model = TransformerModel(vocab_size=30000, n_layer=3, n_head=4, n_emb=8, context_length=8, pad_token_id=0)
checkpoint = torch.load(CHECK_POINT)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
input_data = input("input ids :")

# Convert input string to a list of integers
input_ids_list = [int(x) for x in input_data.split(",")]
input_tensor = torch.tensor([input_ids_list], dtype=torch.long)

# Calculate how much padding is needed to reach 256 tokens
padding_needed = max(0, 8 - input_tensor.shape[1])

# Pad the input_tensor to have a sequence length of 256
input_tensor_padded = F.pad(input_tensor, (0, padding_needed), 'constant', 0)

# Now you can pass the input_tensor to the model
with torch.no_grad():
    output = model(input_tensor_padded)

print("output ids below")
print(output.shape)
print(output)
# Assuming `output` is the matrix of probabilities from your model
# Assuming output_tensor is your model output with shape [1, 256, 30000]
_, predicted_token_ids = torch.max(output, dim=2)

# Print the shape and the predicted token IDs to verify
print(predicted_token_ids.shape)
print(predicted_token_ids)

print("Token IDs:", predicted_token_ids)
