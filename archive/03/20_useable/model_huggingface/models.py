import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel

class DecoderOnlyTransformer(GPT2LMHeadModel):
    def __init__(self, vocab_size, decoder_layer, n_head, n_emb, context_length, pad_token_id):
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=context_length,
            n_ctx=context_length,
            n_embd=n_emb,
            n_layer=decoder_layer,
            n_head=n_head,
            pad_token_id=pad_token_id
        )
        super().__init__(config)