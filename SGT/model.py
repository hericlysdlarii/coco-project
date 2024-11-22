import torch.nn as nn
# import segmentation_models_pytorch as smp
import torchvision.models as models
import torch

# Extrator de características da imagem usando ResNet
class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])  # Remove a última camada FC

        # Camada linear para reduzir a dimensão de 2048 para 512
        self.fc = nn.Linear(2048, 512)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten para (batch_size, 2048)
        features = self.fc(features)  # Reduz para (batch_size, 512)
        return features


# Decodificador LSTM para geração de texto
class CaptionGenerator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CaptionGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # Dropout na LSTM
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # Concatena feature com legendas
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs