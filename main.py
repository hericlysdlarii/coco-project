from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import SGT
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

# from SGT import model

if __name__ == '__main__':

    # Configuração de dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parâmetros do modelo
    BATCH_SIZE = 3
    EPOCHS = 1
    embed_size = 512
    hidden_size = 224
    vocab_size = 50000 
    

    dataloader = SGT.Dataloader(batch_size=BATCH_SIZE, size=hidden_size, shuffle=True)

    train_dataloader = dataloader.get_train_dataloader()
    val_dataloader = dataloader.get_val_dataloader()
    # test_dataloader = dataloader.get_test_dataloader()

    # print(type(train_dataloader))

    feature_extractor = SGT.ImageFeatureExtractor().to(device)
    caption_generator = SGT.CaptionGenerator(embed_size, hidden_size, vocab_size).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignora o índice 0, se usado para padding
    optimizer = optim.Adam(caption_generator.parameters(), lr=0.001)

    running_loss = []
    val_running_loss = []

    for epoch in range(0, EPOCHS):
        feature_extractor.eval()  # Congela o extrator de características
        caption_generator.train()

        running_loss = []
        train_samples = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, captions in train_samples:
            images, captions = images.to(device), captions.to(device)

            # Extrair características da imagem sem gradiente
            with torch.no_grad():
                features = feature_extractor(images) 
                # print(features.shape)  

            # Prepara entradas e alvos
            inputs = captions[:, :-2]  # Remove a última palavra
            targets = captions[:, 1:]  # Remove a primeira palavra

            # print('Features', features.shape)
            # print('Inputs', inputs.shape)

            optimizer.zero_grad()
            outputs = caption_generator(features, inputs)

            # Ajusta a forma de outputs para (batch_size * sequence_length, vocab_size)
            outputs = outputs.reshape(-1, vocab_size)

            # Ajusta a forma de targets para (batch_size * sequence_length)
            targets = targets.reshape(-1)

            # print('Out', outputs.shape)
            # print('Tar', targets.shape)

            # Calcular a perda
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            train_samples.set_description(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {np.mean(running_loss):0.3f}")
            
            # image = image.detach().cpu().numpy()[0].transpose(1, 2, 0)
            
            # cv2.imshow('Image', image)

        caption_generator.eval()
        with torch.no_grad():
            val_samples = tqdm(val_dataloader, desc="Validation")
            for images, captions in val_samples:
                images, captions = images.to(device), captions.to(device)

                # Extrair características da imagem
                features = feature_extractor(images)

                # Prepara entradas e alvos
                inputs = captions[:, :-2]
                targets = captions[:, 1:]

                # Geração da legenda
                outputs = caption_generator(features, inputs)

                # Ajusta as formas
                outputs = outputs.reshape(-1, vocab_size)
                targets = targets.reshape(-1)

                # Calcula a perda
                loss = criterion(outputs, targets)
                val_running_loss.append(loss.item())
                val_samples.set_description(f"Validation Loss: {np.mean(val_running_loss):0.3f}")

        print(f"Validation Loss after Epoch {epoch+1}: {np.mean(val_running_loss):0.3f}")

    # # Plotando a atualização do erro
    # plt.figure(figsize=(10, 5))
    # plt.plot(running_loss)
    # plt.xlabel('Épocas')
    # plt.ylabel('Loss')
    # plt.title('Histórico de Loss')
    # plt.show()

                