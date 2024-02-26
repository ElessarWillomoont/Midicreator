from transformers import AutoTokenizer
n_layer=9
n_head=4
n_emb=128
context_length = 1024
tokenizer = AutoTokenizer.from_pretrained("Lancelort5786/MIDIGPT_MAESTRO_tokenizer")
from transformers import AutoConfig, GPT2LMHeadModel

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_positions=context_length,
    n_layer=n_layer,
    n_head=n_head,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    n_embd=n_emb
)

model = GPT2LMHeadModel(config)
model
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")