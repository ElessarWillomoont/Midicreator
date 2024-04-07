import torch
from shared.models import DecoderOnlyTransformer
import torch.nn.functional as F

CHECK_POINT = "model_output/archive/ckpt_loss_not_change.pt"
MAX_LENGTH = 32
PAD_ID = 0

temperature = 0.8  # 控制随机性，较低的值意味着较少的随机性
top_k = 50  # Top-K 抽样
top_p = 0.95  # Nucleus 抽样

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

def adjust_logits_for_structure(output_logits, structure_index, insert_oth=False):
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

# Loop until 128 tokens are generated
while len(generated_tokens) < 128:
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
    output_logits = adjust_logits_for_structure(output_logits, structure_index, insert_oth)

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
