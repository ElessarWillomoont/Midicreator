import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class GPTLIKEtransformer(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_emb, context_length, pad_token_id):
        super(TransformerModel, self).__init__()

        self.n_layer = n_layer
        self.n_head = n_head
        self.n_emb = n_emb
        self.context_length = context_length

        # Token embedding layer
        self.token_embeddings = nn.Embedding(vocab_size, n_emb)
        # Positional encoding layer
        self.position_embeddings = nn.Embedding(context_length, n_emb)
        # Dropout layer
        self.dropout = nn.Dropout(0.1)

        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(
            d_model=n_emb, nhead=n_head, 
            dim_feedforward=4 * n_emb, dropout=0.1
        )
        self.transformer_encoders = TransformerEncoder(encoder_layers, num_layers=n_layer)

        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_emb)

        # Output head to generate logits for next token prediction
        self.head = nn.Linear(n_emb, vocab_size, bias=False)

        self.pad_token_id = pad_token_id

    def forward(self, input_ids, mask=None):
        # Create position ids (0 to context_length - 1) and add token embeddings
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Retrieve token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Sum token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)

        # Prepare attention mask
        if mask is None:
            mask = input_ids == self.pad_token_id
        mask = mask.transpose(0, 1)

        # Transformer encoder layers
        transformer_output = self.transformer_encoders(
            embeddings.transpose(0, 1), src_key_padding_mask=mask
        )

        # Final layer normalization
        transformer_output = self.ln_f(transformer_output)

        # Output logits
        logits = self.head(transformer_output.transpose(0, 1))

        return logits