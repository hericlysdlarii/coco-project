import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


# Codificação Posicional
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)


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


# Transformer Encoder usando camadas do PyTorch
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=2048, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(embed_dim)

    def forward(self, x):
        x = self.positional_encoding(x.unsqueeze(1))  # Adiciona dimensão seq_len
        x = self.encoder(x)
        return x


# Transformer Decoder usando camadas do PyTorch
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, vocab_size):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=2048, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, encoder_output):
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        embeddings = self.positional_encoding(embeddings)  # Adiciona codificação posicional
        embeddings = embeddings.permute(1, 0, 2)  # Seq-first para TransformerDecoder
        encoder_output = encoder_output.permute(1, 0, 2)  # Seq-first para TransformerDecoder

        decoder_output = self.decoder(embeddings, encoder_output)  # [seq_len, batch_size, embed_dim]
        logits = self.fc_out(decoder_output.permute(1, 0, 2))  # [batch_size, seq_len, vocab_size]
        return logits


# Label Smoothing Loss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, preds, target):
        n_classes = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        one_hot = F.one_hot(target, n_classes).float()
        smoothed_labels = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        return torch.mean(-smoothed_labels * log_preds)


# Atualiza o modelo de geração de legendas
class CaptionGenerator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_heads, num_encoder_layers, num_decoder_layers, max_len=20):
        super(CaptionGenerator, self).__init__()
        self.encoder = TransformerEncoder(embed_size, num_heads, num_encoder_layers)
        self.decoder = TransformerDecoder(embed_size, num_heads, num_decoder_layers, vocab_size)
        self.loss_fn = LabelSmoothingLoss()
        self.max_len = max_len  # Comprimento máximo da legenda
        self.vocab_size = vocab_size
        self.start_token_id = 1  # ID do token de início
        self.end_token_id = 2  # ID do token de fim

    def forward(self, features, captions=None, targets=None):
        # Etapa de treinamento
        if captions is not None:
            encoder_output = self.encoder(features)  # [batch_size, seq_len, embed_size]
            outputs = self.decoder(captions, encoder_output)  # [batch_size, seq_len, vocab_size]

            if targets is not None:
                loss = self.loss_fn(outputs.permute(0, 2, 1), targets)  # Ajusta dimensões para CrossEntropy
                return outputs, loss
            return outputs

        # Etapa de inferência
        else:
            batch_size = features.size(0)
            encoder_output = self.encoder(features)  # [batch_size, seq_len, embed_size]

            # Inicializa as legendas com o token de início
            generated_captions = torch.full((batch_size, 1), self.start_token_id, dtype=torch.long).to(features.device)

            for _ in range(self.max_len):
                # Decodifica a sequência gerada até o momento
                decoder_output = self.decoder(generated_captions, encoder_output)  # [batch_size, seq_len, vocab_size]
                next_token = decoder_output[:, -1, :].argmax(dim=-1, keepdim=True)  # Seleciona o token com maior probabilidade

                # Concatena o token previsto à sequência gerada
                generated_captions = torch.cat((generated_captions, next_token), dim=1)

                # Para se encontrar o token de fim em todas as sequências
                if (next_token == self.end_token_id).all():
                    break

            return generated_captions

