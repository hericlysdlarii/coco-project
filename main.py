from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import SGT
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

# from SGT import model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 2
    EPOCHS = 1000
    embed_size = 512
    hidden_size = 224
    vocab_size = 50000

    # Carregar dados e vocabulário
    dataloader = SGT.Dataloader(batch_size=BATCH_SIZE, size=hidden_size, shuffle=True)
    vocab = SGT.Vocab()  # Classe de vocabulário
    train_dataloader = dataloader.get_train_dataloader()
    val_dataloader = dataloader.get_val_dataloader()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    feature_extractor = SGT.ImageFeatureExtractor().to(device)
    caption_generator = SGT.CaptionGenerator(embed_size, hidden_size, vocab_size).to(device)
    metrics = SGT.TextMetrics()
    early_stopping = SGT.EarlyStopping(patience=10, delta=0.001)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(caption_generator.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    for epoch in range(EPOCHS):
        # Treinamento
        caption_generator.train()
        feature_extractor.eval()

        train_running_loss = []
        bleu_train, meteor_train = [], []
        target = []

        train_samples = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, captions in train_samples:
            images, captions = images.to(device), captions.to(device)

            with torch.no_grad():
                features = feature_extractor(images)

            inputs = captions[:, :-2]
            targets = captions[:, 1:]

            optimizer.zero_grad()
            outputs = caption_generator(features, inputs)

            # Decodifique as saídas do modelo
            predicted_indices = torch.argmax(outputs, dim=-1)
            predicted_text = tokenizer.decode(predicted_indices[0].cpu().numpy().tolist(), skip_special_tokens=True)

            targets_text = tokenizer.decode(targets[0].cpu().numpy(), skip_special_tokens=True)

            # print('Target: ', targets_text)
            # print('Output: ', predicted_text)

            # tokenized_references = [references.split() for references in targets_text]
            # tokenized_hypotheses = [hypothesis.split() for hypothesis in predicted_text]

            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_running_loss.append(loss.item())

            # bleu, meteor = metrics.evaluate(tokenized_references, tokenized_hypotheses)

            # bleu_train.append(bleu)
            # meteor_train.append(meteor)

            train_samples.set_description(
                f"Epoch {epoch+1}/{EPOCHS} | Loss: {np.mean(train_running_loss):0.3f} | BLEU: {np.mean(bleu_train):0.3f} | METEOR: {np.mean(meteor_train):0.3f}"
            )

        # Validação
        caption_generator.eval()
        val_running_loss, bleu_val, meteor_val = [], [], []

        target, output = [], []

        val_samples = tqdm(val_dataloader, desc="Validation")
        with torch.no_grad():
            for images, captions in val_samples:
                images, captions = images.to(device), captions.to(device)
                features = feature_extractor(images)

                inputs = captions[:, :-2]
                targets = captions[:, 1:]

                outputs = caption_generator(features, inputs)

                # Decodifique as saídas do modelo
                predicted_indices = torch.argmax(outputs, dim=-1)
                predicted_text = tokenizer.decode(predicted_indices[0].cpu().numpy().tolist(), skip_special_tokens=True)

                targets_text = tokenizer.decode(targets[0].cpu().numpy(), skip_special_tokens=True)

                target.append(targets_text)
                output.append(predicted_text)

                outputs = outputs.reshape(-1, vocab_size)
                targets = targets.reshape(-1)

                loss = criterion(outputs, targets)
                val_running_loss.append(loss.item())

                # outputs_text = tokenizer.decode(outputs.tolist(), skip_special_tokens=True)
                # targets_text = tokenizer.decode(targets.tolist(), skip_special_tokens=True)
                # bleu, meteor = metrics.evaluate(outputs_text, targets_text)
                # bleu_val.append(bleu)
                # meteor_val.append(meteor)

                val_samples.set_description(
                    f"Epoch {epoch+1}/{EPOCHS} | Loss: {np.mean(val_running_loss):0.3f} | BLEU: {np.mean(bleu_val):0.3f} | METEOR: {np.mean(meteor_val):0.3f}"
                )

        print('Target from last bath val: ', target[(len(val_samples)-1)])
        print('Output from last bath val: ', output[(len(val_samples)-1)])
        

        scheduler.step(np.mean(val_running_loss))
        early_stopping(np.mean(val_running_loss))
        if early_stopping.early_stop():
            print("Early Stopping")
            break


    # # Plotando a atualização do erro
    # plt.figure(figsize=(10, 5))
    # plt.plot(running_loss)
    # plt.xlabel('Épocas')
    # plt.ylabel('Loss')
    # plt.title('Histórico de Loss')
    # plt.show()

                