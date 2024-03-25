import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, encoder_layer, decoder_layer, n_head, n_emb, context_length, pad_token_id):
        super(TransformerModel, self).__init__()

        self.pad_token_id = pad_token_id

        # Token embedding layer
        self.token_embeddings = nn.Embedding(vocab_size, n_emb)
        # Positional encoding layer for encoder
        self.position_embeddings_enc = nn.Embedding(context_length, n_emb)
        # Positional encoding layer for decoder, only initialized if decoder_layer > 0
        self.position_embeddings_dec = nn.Embedding(context_length, n_emb) if decoder_layer > 0 else None
        # Dropout layer
        self.dropout = nn.Dropout(0.1)

        # Transformer model, conditional initialization
        if encoder_layer > 0 and decoder_layer > 0:
            self.use_decoder = True
            self.transformer = nn.Transformer(
                d_model=n_emb, 
                nhead=n_head, 
                num_encoder_layers=encoder_layer, 
                num_decoder_layers=decoder_layer, 
                dim_feedforward=4 * n_emb, 
                dropout=0.1,
                batch_first=True
            )
        elif encoder_layer > 0:
            self.use_decoder = False
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb, dropout=0.1),
                num_layers=encoder_layer
            )
        elif decoder_layer > 0:
            self.use_decoder = True
            self.transformer = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb, dropout=0.1),
                num_layers=decoder_layer
            )
        else:
            raise ValueError("Both encoder_layer and decoder_layer cannot be 0.")

        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_emb)

        # Output head to generate logits for next token prediction
        self.head = nn.Linear(n_emb, vocab_size, bias=False)

    def forward(self, input_ids, src_mask=None):
        position_ids_enc = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids_enc = position_ids_enc.unsqueeze(0).expand_as(input_ids)
        
        token_embeddings_enc = self.token_embeddings(input_ids)
        position_embeddings_enc = self.position_embeddings_enc(position_ids_enc)
        embeddings_enc = token_embeddings_enc + position_embeddings_enc
        embeddings_enc = self.dropout(embeddings_enc)
        
        if src_mask is None:
            src_mask = input_ids == self.pad_token_id

        # Adjust forward pass based on the transformer model configuration
        if hasattr(self, 'use_decoder') and self.use_decoder:
            if isinstance(self.transformer, nn.Transformer):
                dummy_tgt = torch.zeros_like(embeddings_enc)
                transformer_output = self.transformer(
                    src=embeddings_enc,
                    tgt=dummy_tgt,
                    src_key_padding_mask=src_mask
                )
            elif isinstance(self.transformer, nn.TransformerDecoder):
                # For simplicity, using embeddings as memory; replace with actual encoder output as needed
                memory = embeddings_enc
                dummy_tgt = torch.zeros_like(embeddings_enc)
                transformer_output = self.transformer(
                    tgt=dummy_tgt,  # Assuming a dummy target for illustration; adjust as needed
                    memory=memory,
                    tgt_key_padding_mask=src_mask  # Example mask usage; adjust as needed
                )
        else:  # Pure encoder
            transformer_output = self.transformer(
                src=embeddings_enc,
                src_key_padding_mask=src_mask
            )

        transformer_output = self.ln_f(transformer_output)
        logits = self.head(transformer_output)

        return logits