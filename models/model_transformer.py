import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_emb, context_length, pad_token_id):
        super(TransformerModel, self).__init__()

        self.pad_token_id = pad_token_id

        # Token embedding layer
        self.token_embeddings = nn.Embedding(vocab_size, n_emb)
        # Positional encoding layer
        self.position_embeddings = nn.Embedding(context_length, n_emb)
        # Dropout layer
        self.dropout = nn.Dropout(0.1)

        # Transformer model
        self.transformer = nn.Transformer(
            d_model=n_emb, 
            nhead=n_head, 
            num_encoder_layers=n_layer, 
            num_decoder_layers=0,  # 不使用解码器层
            dim_feedforward=4 * n_emb, 
            dropout=0.1,
            batch_first=True  # 启用 batch_first
        )

        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_emb)

        # Output head to generate logits for next token prediction
        self.head = nn.Linear(n_emb, vocab_size, bias=False)

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
        # Convert mask to expected format for nn.Transformer
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = (1.0 - mask.float()) * -10000.0  # Convert to float mask

        # Transformer encoder
        transformer_output = self.transformer(
            src=embeddings, 
            src_key_padding_mask=mask
        )

        # Final layer normalization
        transformer_output = self.ln_f(transformer_output)

        # Output logits
        logits = self.head(transformer_output)

        return logits
