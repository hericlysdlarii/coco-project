import json
import os
import random
import cv2
from glob import glob
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Data(Dataset):
    def __init__(self, image_dir: str, image_split: str, caption_split: str, transform=None) -> None:
        self._image_dir = image_dir
        self._image_paths = glob(f'{image_dir}/{image_split}/*.jpg')
        self._caption_file = f'{image_dir}/{caption_split}/val_captions.json'
        self._transform = transform

        # Carrega as legendas uma vez, ao inicializar a classe
        with open(self._caption_file, 'r') as f:
            self._captions = json.load(f)


    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> tuple:
        image_path = self._image_paths[idx]
        
        # Carrega a imagem
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Obtém o ID da imagem para recuperar a legenda correspondente
        image_id = str(int(os.path.basename(image_path).split('.')[0]))
        
        image_captions = [anno['captions'] for anno in self._captions if str(int(anno['image_id'])) == image_id]
        
        # Aplica transformações se houver
        if self._transform:
            augmented = self._transform(image=image)
            image = augmented['image']

        # Retorna a imagem e a(s) legenda(s)
        return image, image_captions


class Dataloader:
    def __init__(self, batch_size: int, shuffle: bool, size: int, subset: int = 0, description: bool = False) -> None:
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._subset = subset
        self._dir = 'coco2017'
        self._description = description
        self._size = size
        self._prob_aug = 0.3
        #self._transform = self.compose()

    # def compose(self):

    #     trans_train = A.Compose([
    #         A.Resize(height=self._size, width=self._size),
    #         ToTensorV2(),
    #     ])

    #     trans_val = A.Compose([
    #         A.Resize(height=self._size, width=self._size),
    #         ToTensorV2(),
    #     ])

    #     return {
    #         'train2017': trans_train,
    #         'val2017': trans_val,
    #     }
    
    def _transform(self) -> A.Compose:
        return A.Compose([
           
            A.Resize(height=self._size, width=self._size),
            A.RandomBrightnessContrast(p=self._prob_aug),
            
            ToTensorV2(),

        ])

    def get_dataloader(self, image_split: str, caption_split: str) -> DataLoader:
        transform = self._transform()
        dataset = Data(self._dir, image_split, caption_split, transform)

        if self._subset:
            dataset = Subset(dataset, range(self._subset))

        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=self._shuffle)
        
        if self._description:
            return tqdm(dataloader)
        
        return dataloader


    def get_train_dataloader(self) -> DataLoader: return self.get_dataloader('val2017', 'captions')
    # def get_val_dataloader(self) -> DataLoader: return self.get_dataloader('val')
    # def get_test_dataloader(self) -> DataLoader: return self.get_dataloader('test')


