from transformers import GPT2LMHeadModel

model_path = "output/checkpoint-140000"  
model = GPT2LMHeadModel.from_pretrained(model_path)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Lancelort5786/MIDIGPT_MAESTRO_tokenizer")

prompt = input("Your text prompt here")
inputs = tokenizer(prompt, return_tensors="pt")

# Generate tex=
output_sequences = model.generate(
    input_ids=inputs['input_ids'], 
    max_length=1024,  # Adjust the max_length accordingly
    num_return_sequences=1,  # Number of sequences to generate
    no_repeat_ngram_size=2,  # This helps in reducing repetition
    temperature=1.0,  # Controls randomness. Lower is less random.
    top_k=50,  # Top-K sampling
    top_p=0.95,  # Nucleus sampling
    # Add other parameters as needed
)

# Decode the output sequences to text
generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(generated_text)
