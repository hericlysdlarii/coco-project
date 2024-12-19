import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torch
import torch.nn.functional as F


# Extrator de características da imagem usando ResNet
class ImageFeatureExtractor(nn.Module):
    def __init__(self, embed_size):
        super(ImageFeatureExtractor, self).__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])  # Remove a última camada FC
        self.feature_projection = nn.Linear(2048, embed_size)  # Projeta para embed_size dinâmico

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten para [batch_size, 2048]
        features = self.feature_projection(features)  # Projeta para [batch_size, embed_size]
        return features


# Transformer Encoder Layer em PyTorch
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.dense = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Normaliza e passa pela camada densa
        x_norm = self.layer_norm_1(x)
        x_dense = self.dense(x_norm)

        # Atenção multi-cabeças
        attn_output, _ = self.attention(x_dense, x_dense, x_dense)
        x = x + attn_output  # Residual Connection
        x = self.layer_norm_2(x)
        return x


# Transformer Decoder Layer em PyTorch
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, units, num_heads, vocab_size):
        super(TransformerDecoderLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.attention_1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.attention_2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.layer_norm_3 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, units),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(units, embed_dim),
            nn.Dropout(0.5)
        )
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, encoder_output, mask=None):
        # Embedding e máscara causal
        embeddings = self.embedding(input_ids).permute(1, 0, 2)  # Seq first para MultiheadAttention
        causal_mask = self.generate_causal_mask(embeddings.size(0), embeddings.device)

        # Primeira camada de atenção (self-attention)
        attn_output_1, _ = self.attention_1(embeddings, embeddings, embeddings, attn_mask=causal_mask)
        out_1 = self.layer_norm_1(embeddings + attn_output_1)

        # Segunda camada de atenção (encoder-decoder attention)
        attn_output_2, _ = self.attention_2(out_1, encoder_output, encoder_output)
        out_2 = self.layer_norm_2(out_1 + attn_output_2)

        # Feed Forward Network
        ffn_output = self.ffn(out_2)
        out = self.layer_norm_3(out_2 + ffn_output)

        # Predições finais
        preds = self.out(out.permute(1, 0, 2))  # Volta para batch first
        return preds

    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))


# Atualiza o decodificador existente no CaptionGenerator
class CaptionGenerator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_heads):
        super(CaptionGenerator, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(embed_size, num_heads)
        self.decoder_layer = TransformerDecoderLayer(embed_size, hidden_size, num_heads, vocab_size)

    def forward(self, features, captions):
        # Encoder
        encoder_output = self.encoder_layer(features.unsqueeze(0))  # Adiciona dimensão seq_len

        # Decoder
        outputs = self.decoder_layer(captions, encoder_output)
        return outputs


# # Decodificador LSTM para geração de texto
# class CaptionGenerator(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout_prob=0.5):
#         super(CaptionGenerator, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True) 
#         self.dropout = nn.Dropout(p=dropout_prob)
#         self.linear = nn.Linear(hidden_size, vocab_size)

#     def forward(self, features, captions):
#         features = features.view(features.size(0), -1)  # [8, 2048]
#         features = features.unsqueeze(1) # Expande para [8, 1, 2048]

#         embeddings = self.embedding(captions)

#         embeddings = torch.cat((features, embeddings), dim=1)  # Concatena feature com legendas
#         hiddens, _ = self.lstm(embeddings)
#         hiddens = self.dropout(hiddens)
#         outputs = self.linear(hiddens)
#         return outputs