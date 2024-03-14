import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import EncoderDecoderModel, BertConfig

class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, encoder_layer, n_head, n_emb, context_length, pad_token_id):
        super().__init__()
        self.config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=n_emb,
            num_attention_heads=n_head,
            num_hidden_layers=encoder_layer,
            max_position_embeddings=context_length,
            pad_token_id=pad_token_id
        )
        self.encoder = BertEncoder(self.config)
        self.embeddings = nn.Embedding(vocab_size, n_emb)
        self.head = nn.Linear(n_emb, vocab_size)

    def forward(self, input_ids):
        extended_attention_mask = torch.ones(input_ids.shape, device=input_ids.device)  # Simple mask, adjust as needed
        embedding_output = self.embeddings(input_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]
        logits = self.head(sequence_output)
        return logits

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

class CustomEncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, encoder_layer, decoder_layer, n_head, n_emb, context_length, pad_token_id):
        super().__init__()
        encoder_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=n_emb,
            num_attention_heads=n_head,
            num_hidden_layers=encoder_layer,
            max_position_embeddings=context_length,
            pad_token_id=pad_token_id
        )
        decoder_config = encoder_config.copy()
        decoder_config.num_hidden_layers = decoder_layer
        self.model = EncoderDecoderModel(config_encoder=encoder_config, config_decoder=decoder_config)

    def forward(self, input_ids, decoder_input_ids=None, labels=None):
        outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
        return outputs