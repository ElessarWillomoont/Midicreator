import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_emb, context_length, pad_token_id, use_decoder=True):
        super(TransformerModel, self).__init__()

        self.pad_token_id = pad_token_id
        self.use_decoder = use_decoder  # Flag to toggle decoder usage

        # Token embedding layer
        self.token_embeddings = nn.Embedding(vocab_size, n_emb)
        # Positional encoding layer for encoder
        self.position_embeddings_enc = nn.Embedding(context_length, n_emb)
        # Optionally, Positional encoding layer for decoder (if different from encoder)
        self.position_embeddings_dec = nn.Embedding(context_length, n_emb) if use_decoder else None
        # Dropout layer
        self.dropout = nn.Dropout(0.1)

        # Transformer model
        self.transformer = nn.Transformer(
            d_model=n_emb, 
            nhead=n_head, 
            num_encoder_layers=n_layer, 
            num_decoder_layers=n_layer if use_decoder else 0,  # Enable decoder layers if use_decoder is True
            dim_feedforward=4 * n_emb, 
            dropout=0.1,
            batch_first=True  # Enable batch_first
        )

        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_emb)

        # Output head to generate logits for next token prediction
        self.head = nn.Linear(n_emb, vocab_size, bias=False)
        
        # List to store intermediate outputs
        self.intermediate_outputs = []

        # Register hooks for each layer
        for layer in self.transformer.modules():
            if isinstance(layer, nn.MultiheadAttention):
                self.register_forward_hook(layer)

    def forward(self, input_ids, src_mask=None):
        # Generate position IDs for encoder
        position_ids_enc = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids_enc = position_ids_enc.unsqueeze(0).expand_as(input_ids)

        # Obtain token embeddings from the token_embeddings layer
        token_embeddings_enc = self.token_embeddings(input_ids)
        # Generate position embeddings for encoder
        position_embeddings_enc = self.position_embeddings_enc(position_ids_enc)

        # Sum token and position embeddings for encoder
        embeddings_enc = token_embeddings_enc + position_embeddings_enc
        embeddings_enc = self.dropout(embeddings_enc)

        # Prepare encoder attention mask
        if src_mask is None:
            src_mask = input_ids == self.pad_token_id

        if self.use_decoder:
            # Create a dummy target tensor for decoder
            dummy_tgt = torch.zeros_like(embeddings_enc)
            # Transformer encoder-decoder
            transformer_output = self.transformer(
                src=embeddings_enc, 
                tgt=dummy_tgt,  # Include dummy target tensor
                src_key_padding_mask=src_mask  # Correctly pass the src_mask without additional modifications
            )
        else:
            # Transformer encoder only
            transformer_output = self.transformer.encoder(
                src=embeddings_enc,
                src_key_padding_mask=src_mask
            )
        
        # Final layer normalization
        transformer_output = self.ln_f(transformer_output)
        logits = self.head(transformer_output)

        return logits, self.intermediate_outputs

    def register_forward_hook(self, layer):
        """Register hook to store intermediate outputs."""
        def hook(module, input, output):
            self.intermediate_outputs.append(output.detach().cpu().numpy())
        layer.register_forward_hook(hook)

    def _generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
