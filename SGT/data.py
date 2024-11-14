import json
import os
import cv2
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn.utils.rnn import pad_sequence
# from torchvision import transforms

from transformers import AutoTokenizer



class Data(Dataset):
    def __init__(self, image_dir: str, image_split: str, caption_split: str, transform=None) -> None:
        self._image_dir = image_dir
        self._image_paths = glob(f'{image_dir}/{image_split}/*.jpg')
        self._caption_file_train = f'{image_dir}/{caption_split}/train_captions.json'
        self._caption_file_val = f'{image_dir}/{caption_split}/val_captions.json'
        self._transform = transform

        # Inicializa o tokenizador uma vez durante a inicialização
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Carrega as legendas de acordo com a divisão de dados (train ou val)
        try:
            if image_split == 'train':
                with open(self._caption_file_train, 'r') as f:
                    self._captions = json.load(f)
            else:
                with open(self._caption_file_val, 'r') as f:
                    self._captions = json.load(f)
        except FileNotFoundError:
            print(f"Caption file for {image_split} not found.")
            self._captions = {}  # Usa um dicionário vazio como fallback

    def __len__(self) -> int:
        return len(self._image_paths)
    

    def __getitem__(self, idx: int) -> tuple:
        image_path = self._image_paths[idx]
        
        # Carrega a imagem com tratamento de erros
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Obtém o ID da imagem para recuperar a legenda correspondente
        image_id = str(int(os.path.basename(image_path).split('.')[0]))
        
        # Tenta recuperar a legenda correspondente ao ID da imagem
        image_captions = [anno['captions'] for anno in self._captions if str(int(anno['image_id'])) == image_id]
        
        if not image_captions:
            print(f"No captions found for image ID {image_id}. Returning an empty caption.")
            caption = "<empty>"
        else:
            # Usa a primeira legenda disponível
            caption = image_captions[0]
            # print(f"Caption for image {image_id}: {caption}")

        
        tokens = self.tokenizer(str(caption), return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        caption_ids = tokens["input_ids"].squeeze()

        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)

        if self._transform:
            augmented = self._transform(image=image)
            image = augmented['image']

        return image, caption_tensor
    

class MyPreProcessing:
    def __init__(self):
        self.available_keys = set()

    def __call__(self, image):
        image = (image/255.).astype('float32')
        return {'image': image}


class Dataloader:
    def __init__(self, batch_size: int, size: int, shuffle: bool, subset: int = 0) -> None:
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._subset = subset
        self._dir = 'coco2017'
        self._size = size
        self._prob_aug = {
            'train': 0.1,
            'val': 0.1,
            'test': 0.,
        }
    
    def _transform(self, image_split: str) -> A.Compose:
        p = self._prob_aug[image_split]
        
        return A.Compose([
           
            A.Resize(height=self._size, width=self._size),
            A.RandomBrightnessContrast(p=p),

            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),

            MyPreProcessing(),
            ToTensorV2(),

        ])
    
    def _collate_fn(self, batch):
        images, captions = zip(*batch)  # Separa imagens e legendas
        images = torch.stack(images, 0)  # Empilha as imagens no batch

        # Adiciona padding nas legendas para que tenham o mesmo comprimento
        captions = pad_sequence(captions, batch_first=True, padding_value=0)  # 0 para token de padding

        return images, captions

    def get_dataloader(self, image_split: str, caption_split: str=None) -> DataLoader:
        dataset = Data(self._dir, image_split, caption_split, self._transform(image_split))

        if self._subset:
            dataset = Subset(dataset, range(self._subset))

        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=self._shuffle, collate_fn=self._collate_fn) #
        
        return dataloader


    def get_train_dataloader(self) -> DataLoader: return self.get_dataloader('train', 'captions')
    def get_val_dataloader(self) -> DataLoader: return self.get_dataloader('val', 'captions')
    # def get_test_dataloader(self) -> DataLoader: return self.get_dataloader('test')


