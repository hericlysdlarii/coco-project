from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import SGT
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer

import ast
import re

# from SGT import model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    EPOCHS = 1000
    embed_size = 512
    hidden_size = 224
    vocab_size = 50000
    weight_decay = 0.01

    # Carregar dados e vocabulário
    dataloader = SGT.Dataloader(batch_size=BATCH_SIZE, size=hidden_size, shuffle=True)
    vocab = SGT.Vocab()  # Classe de vocabulário
    train_dataloader = dataloader.get_train_dataloader()
    val_dataloader = dataloader.get_val_dataloader()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    feature_extractor = SGT.ImageFeatureExtractor().to(device)
    caption_generator = SGT.CaptionGenerator(embed_size, hidden_size, vocab_size).to(device)
    metrics = SGT.TextMetrics()
    early_stopping = SGT.EarlyStopping(patience=20, delta=0.001)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(caption_generator.parameters(), lr=0.001, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    
    train_running_loss = []
    bleu_train, rouge_train = [], []
    val_running_loss, bleu_val = [], []

    rouge1_scores_train = []
    rouge2_scores_train = []
    rougeL_scores_train = []

    rouge1_scores_val = []
    rouge2_scores_val = []
    rougeL_scores_val = []

    for epoch in range(EPOCHS):
        # Treinamento
        caption_generator.train()
        feature_extractor.eval()

        train_samples = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, captions in train_samples:
            images, captions = images.to(device), captions.to(device)

            with torch.no_grad():
                features = feature_extractor(images)

            inputs = captions[:, :-1]
            targets = captions[:, :]

            # print(inputs.shape)
            # print(targets.shape)

            optimizer.zero_grad()

            outputs = caption_generator(features, inputs)

            # Decodificando as saídas do modelo
            predicted_indices = torch.argmax(outputs, dim=-1)
            predicted_text = tokenizer.decode(predicted_indices[0].cpu().numpy().tolist(), skip_special_tokens=True)

            targets_text = tokenizer.decode(targets[0].cpu().numpy(), skip_special_tokens=True)

            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_running_loss.append(loss.item())

            # # if epoch >= 1:
            bleu, rouge_tuple = metrics.evaluate(targets_text, predicted_text)
            # print(rouge)
            rouge = rouge_tuple[0]

            rouge1_scores_train.append(rouge['rouge1'].fmeasure)
            rouge2_scores_train.append(rouge['rouge2'].fmeasure)
            rougeL_scores_train.append(rouge['rougeL'].fmeasure)

            # mean_rouge = (sum(rougeL_scores_train) / len(rougeL_scores_train))

            bleu_train.append(bleu)

            train_samples.set_description(
                f"Epoch {epoch+1}/{EPOCHS} | Loss Train: {np.mean(train_running_loss):0.3f} | BLEU Train: {np.mean(bleu_train):0.3f} | ROUGE Train: {np.mean(rougeL_scores_train):0.3f}"
            )

        # Validação
        caption_generator.eval()

        target, output = [], []

        val_samples = tqdm(val_dataloader, desc="Validation")
        with torch.no_grad():
            for images, captions in val_samples:
                images, captions = images.to(device), captions.to(device)
                features = feature_extractor(images)

                inputs = captions[:, :-1]
                targets = captions[:, :]

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

                bleu, rouge_tuple = metrics.evaluate(predicted_text, targets_text)
                
                rouge = rouge_tuple[0]

                rouge1_scores_val.append(rouge['rouge1'].fmeasure)
                rouge2_scores_val.append(rouge['rouge2'].fmeasure)
                rougeL_scores_val.append(rouge['rougeL'].fmeasure)

                bleu_val.append(bleu)

                val_samples.set_description(
                    f"Epoch {epoch+1}/{EPOCHS} | Loss Val: {np.mean(val_running_loss):0.3f} | BLEU Val: {np.mean(bleu_val):0.3f} | ROUGE Val: {np.mean(rougeL_scores_val):0.3f}"
                )

        print('Target from last bath val: ', target[(len(val_samples)-1)])
        print('Output from last bath val: ', output[(len(val_samples)-1)])
        print(' ')
        
        # print('LR atual: {}'.format(scheduler.get_lr()))
        scheduler.step(np.mean(val_running_loss))
        # lr = scheduler.get_lr()
        # print('Lr: ', lr)
        early_stopping(np.mean(val_running_loss))
        if early_stopping.early_stop():
            print("Early Stopping")
            break


    # Plotando a atualização do erro
    plt.figure(figsize=(10, 5))
    plt.plot(train_running_loss, label='Training Loss')
    plt.plot(val_running_loss, label='Validation Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.title('Histórico de Loss')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(bleu_train, label='Training Loss')
    plt.plot(bleu_val, label='Validation Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.title('Histórico de Loss')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(rougeL_scores_train, label='Training Loss')
    plt.plot(rougeL_scores_val, label='Validation Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.title('Histórico de Loss')
    plt.show()

                