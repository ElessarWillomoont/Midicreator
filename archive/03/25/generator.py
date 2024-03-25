import torch
from models import DecoderOnlyTransformer
import torch.nn.functional as F

CHECK_POINT = "model_output/archive/ckpt_loss_not_change.pt"
MAX_LENGTH = 32
PAD_ID = 0

temperature = 0.9  # 控制随机性，较低的值意味着较少的随机性
top_k = 50  # Top-K 抽样
top_p = 0.95  # Nucleus 抽样


# Load the model and checkpoint
model = DecoderOnlyTransformer(vocab_size=465, decoder_layer=6, n_head=4, n_emb=768, context_length=MAX_LENGTH, pad_token_id=PAD_ID)
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
    if len(current_input) > MAX_LENGTH:
        current_input = current_input[-MAX_LENGTH:]  # Keep only the last 8 tokens

    # Convert to tensor
    input_tensor = torch.tensor([current_input], dtype=torch.long)
    
    # Create an attention mask for the padded input tensor
    attention_mask = input_tensor != 0
    
    # Model inference
    with torch.no_grad():
        output = model(input_tensor, attention_mask=attention_mask)

    # Access the logits and get the last token predicted
    #output_logits = output.logits
    output_logits = output.logits[:, -1, :] / temperature  # 只处理最后一个时间步的logits，并应用温度调整

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

    # 随机选择一个token
    probabilities = F.softmax(output_logits, dim=-1)
    last_predicted_token = torch.multinomial(probabilities, 1).item()
    
    # Append the predicted token to the list of generated tokens
    generated_tokens.append(last_predicted_token)
    
    # Update the input with the newly generated token
    current_input.append(last_predicted_token)

# Output the generated tokens
print("Generated tokens:", generated_tokens)
