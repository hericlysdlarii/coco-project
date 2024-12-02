import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import SGT

# Configurações globais
CONFIG = {
    "BATCH_SIZE": 8,
    "EPOCHS": 1000,
    "EMBED_SIZE": 512,
    "HIDDEN_SIZE": 224,
    "VOCAB_SIZE": 50000,
    "WEIGHT_DECAY": 0.01,
    "LEARNING_RATE": 0.01,
    "PATIENCE": 10,
    "DELTA": 0.001,
    "CHECKPOINT_PATH": "best_model/best_model.pth",
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Função para treinar uma época
def train_epoch(model, dataloader, feature_extractor, tokenizer, criterion, optimizer, metrics, log, device, vocab_size, epoch):
    model.train()
    feature_extractor.eval()
    running_loss = []
    bleu_scores = []
    rouge_scores = []

    for images, captions in tqdm(dataloader, desc="Training"):
        images, captions = images.to(device), captions.to(device)
        inputs, targets = captions[:, :-1], captions[:, :]

        with torch.no_grad():
            features = feature_extractor(images)

        optimizer.zero_grad()
        outputs = model(features, inputs)

        # Decodifica e avalia uma amostra por lote
        predicted_indices = torch.argmax(outputs, dim=-1)
        predicted_text = tokenizer.decode(predicted_indices[0].cpu().numpy().tolist(), skip_special_tokens=True)
        targets_text = tokenizer.decode(targets[0].cpu().numpy(), skip_special_tokens=True)

        new_outputs = outputs.reshape(-1, vocab_size)
        new_targets = targets.reshape(-1)

        loss = criterion(new_outputs, new_targets)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())

        bleu, rouge_tuple = metrics.evaluate(targets_text, predicted_text)
        rouge = rouge_tuple[0]
        bleu_scores.append(bleu)
        rouge_scores.append(rouge['rougeL'].fmeasure)

    log.log_scalar_train((np.mean(bleu_scores)), epoch=epoch, scalar_name='BLEU')
    log.log_scalar_train((np.mean(rouge_scores)), epoch=epoch, scalar_name='ROUGE')
    log.log_scalar_train((np.mean(running_loss)), epoch=epoch, scalar_name='LOSS')
    log.log_tensors_train(images, targets, outputs, epoch=epoch)

    return {
        "loss": np.mean(running_loss),
        "bleu": np.mean(bleu_scores),
        "rougeL": np.mean(rouge_scores),
    }

# Função para validação
def validate_epoch(model, dataloader, feature_extractor, tokenizer, criterion, metrics, log, device, vocab_size, epoch):
    model.eval()
    running_loss = []
    bleu_scores = []
    rouge_scores = []

    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="Validation"):
            images, captions = images.to(device), captions.to(device)
            inputs, targets = captions[:, :-1], captions[:, :]

            features = feature_extractor(images)
            outputs = model(features, inputs)

            predicted_indices = torch.argmax(outputs, dim=-1)
            predicted_text = tokenizer.decode(predicted_indices[0].cpu().numpy().tolist(), skip_special_tokens=True)
            targets_text = tokenizer.decode(targets[0].cpu().numpy().tolist(), skip_special_tokens=True)

            new_outputs = outputs.reshape(-1, vocab_size)
            new_targets = targets.reshape(-1)

            loss = criterion(new_outputs, new_targets)
            running_loss.append(loss.item())

            bleu, rouge_tuple = metrics.evaluate(targets_text, predicted_text)
            rouge = rouge_tuple[0]
            bleu_scores.append(bleu)
            rouge_scores.append(rouge['rougeL'].fmeasure)
        
        log.log_scalar_val((np.mean(bleu_scores)), epoch=epoch, scalar_name='BLEU')
        log.log_scalar_val((np.mean(rouge_scores)), epoch=epoch, scalar_name='ROUGE')
        log.log_scalar_val((np.mean(running_loss)), epoch=epoch, scalar_name='LOSS')
        log.log_tensors_val(images, targets, outputs, epoch=epoch)

    return {
        "loss": np.mean(running_loss),
        "bleu": np.mean(bleu_scores),
        "rougeL": np.mean(rouge_scores),
    }

# Função para treinar e validar o modelo
def train_and_validate(model, feature_extractor, dataloaders, tokenizer, metrics, log, config):
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, cooldown=20)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    early_stopping = SGT.EarlyStopping(patience=config["PATIENCE"], delta=config["DELTA"])

    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    for epoch in range(config["EPOCHS"]):
        print(f"Epoch {epoch + 1}/{config['EPOCHS']}")

        train_metrics = train_epoch(
            model, dataloaders["train"], feature_extractor, tokenizer, criterion, optimizer, metrics, log, config["DEVICE"], config["VOCAB_SIZE"], epoch
        )
        val_metrics = validate_epoch(
            model, dataloaders["val"], feature_extractor, tokenizer, criterion, metrics, log, config["DEVICE"], config["VOCAB_SIZE"], epoch
        )

        print(f"Train Loss: {train_metrics['loss']:.3f}, BLEU: {train_metrics['bleu']:.3f}, ROUGE-L: {train_metrics['rougeL']:.3f}")
        print(f"Val Loss: {val_metrics['loss']:.3f}, BLEU: {val_metrics['bleu']:.3f}, ROUGE-L: {val_metrics['rougeL']:.3f}")

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Atualizar modelo
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), config["CHECKPOINT_PATH"])
            print(f"Modelo salvo com loss de validação: {best_val_loss:.3f}")

        scheduler.step(val_metrics["loss"])
        early_stopping(val_metrics["loss"])

        # log.log_scalar_hiper(early_stopping.get_fault(), epoch=epoch, scalar_name='FAULT')
        log.log_scalar_hiper(scheduler.get_last_lr()[-1], epoch=epoch, scalar_name='LR')

        if early_stopping.early_stop():
            print("Treinamento interrompido por early stopping.")
            break

    log.close()

    return history

#Função para testar o modelo
def test_model(model, dataloader, feature_extractor, tokenizer, metrics, config):
    model.eval()
    feature_extractor.eval()
    test_running_loss = []
    bleu_scores = []
    rouge_scores = []
    images_list, captions_list, outputs_list = [], [], []

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="Testing"):
            images, captions = images.to(config["DEVICE"]), captions.to(config["DEVICE"])
            inputs, targets = captions[:, :-1], captions[:, :]

            features = feature_extractor(images)
            outputs = model(features, inputs)

            # Decodificar saída do modelo
            predicted_indices = torch.argmax(outputs, dim=-1)
            predicted_text = tokenizer.decode(predicted_indices[0].cpu().numpy().tolist(), skip_special_tokens=True)
            targets_text = tokenizer.decode(targets[0].cpu().numpy().tolist(), skip_special_tokens=True)

            outputs = outputs.reshape(-1, config["VOCAB_SIZE"])
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            test_running_loss.append(loss.item())

            images_list.append(images[0].cpu().numpy().transpose(1, 2, 0))
            captions_list.append(targets_text)
            outputs_list.append(predicted_text)

            bleu, rouge_tuple = metrics.evaluate(targets_text, predicted_text)
            rouge = rouge_tuple[0]
            bleu_scores.append(bleu)
            rouge_scores.append(rouge['rougeL'].fmeasure)

    # Resultados médios
    avg_loss = np.mean(test_running_loss)
    avg_bleu = np.mean(bleu_scores)
    avg_rouge = np.mean(rouge_scores)

    print(f"Test Loss: {avg_loss:.3f}, BLEU: {avg_bleu:.3f}, ROUGE-L: {avg_rouge:.3f}")

    # Visualizar resultados de uma amostra
    plt.figure()
    plt.imshow(images_list[-1])
    plt.title(f"Target: {captions_list[-1]}\n\nPredicted: {outputs_list[-1]}")
    plt.axis("off")
    plt.show()

    return {"loss": avg_loss, "bleu": avg_bleu, "rougeL": avg_rouge}

# #Função para testar o modelo
# def test_model_without_targets(model, dataloader, feature_extractor, tokenizer, config):
#     model.eval()
#     feature_extractor.eval()

#     images_list, outputs_list = [], []

#     with torch.no_grad():
#         for images in tqdm(dataloader, desc="Testing"):
#             images = images.to(config["DEVICE"])

#             # Extração de características
#             features = feature_extractor(images)

#             # Geração de legendas
#             outputs = model(features)

#             # Decodificar saída do modelo
#             predicted_indices = torch.argmax(outputs, dim=-1)
#             predicted_text = tokenizer.decode(predicted_indices[0].cpu().numpy().tolist(), skip_special_tokens=True)

#             images_list.append(images[0].cpu().numpy().transpose(1, 2, 0))
#             outputs_list.append(predicted_text)

#     # Exibir a última imagem e a legenda gerada
#     if images_list and outputs_list:
#         plt.figure()
#         plt.imshow(images_list[-1])
#         plt.title(f"Predicted: {outputs_list[-1]}")
#         plt.axis("off")
#         plt.show()
#     else:
#         print("Nenhuma imagem processada no dataloader de teste.")

#     return {"outputs": outputs_list}


# Função principal
if __name__ == '__main__':
    # Inicializar componentes
    dataloader = SGT.Dataloader(batch_size=CONFIG["BATCH_SIZE"], size=CONFIG["HIDDEN_SIZE"], shuffle=True)
    dataloaders = {
        "train": dataloader.get_train_dataloader(),
        "val": dataloader.get_val_dataloader(),
        "test": dataloader.get_test_dataloader(),
    }
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    feature_extractor = SGT.ImageFeatureExtractor().to(CONFIG["DEVICE"])
    caption_generator = SGT.CaptionGenerator(
        CONFIG["EMBED_SIZE"], 
        CONFIG["HIDDEN_SIZE"], 
        CONFIG["VOCAB_SIZE"], 
    ).to(CONFIG["DEVICE"])
    metrics = SGT.TextMetrics()
    logs = SGT.Log(CONFIG["BATCH_SIZE"], tokenizer)

    # # Treinamento e validação
    history = train_and_validate(caption_generator, feature_extractor, dataloaders, tokenizer, metrics, logs, CONFIG)

    # Plotar histórico
    plt.figure(figsize=(10, 5))
    train_loss = [x["loss"] for x in history["train"]]
    val_loss = [x["loss"] for x in history["val"]]
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

    # Teste
    print("Iniciando a fase de testes...")
    caption_generator.load_state_dict(torch.load(CONFIG["CHECKPOINT_PATH"]))

    test_metrics = test_model(caption_generator, dataloaders["test"], feature_extractor, tokenizer, metrics, CONFIG)

    #Resultados do teste
    print(f"Resultados do Teste - Loss: {test_metrics['loss']:.3f}, BLEU: {test_metrics['bleu']:.3f}, ROUGE-L: {test_metrics['rougeL']:.3f}")
