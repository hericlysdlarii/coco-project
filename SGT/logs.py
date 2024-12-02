from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import git
import datetime
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

class Stamp:
  def __init__(self, repo_path: str = "C:/Users/hericlys.borges/Documents/Mestrado/Projeto/coco-project") -> None:
    self.__repo = git.Repo(repo_path)
    self._commit = self.__repo.head.commit
    self._time = datetime.datetime.now().strftime("%d%m%Y_%H%M")
    self.max_commit_character = 8

  def timestamp(self):
    return self._time
  
  def get_hex(self):
    return self._commit.hexsha[:self.max_commit_character]

  def get_details(self):
    return f'Message: {self._commit.message} | Autor:{self._commit.author} | Data do Commit: {self._commit.committed_datetime}'

class Log:
  def __init__(self, batch_size: int, tokenizer, comment: str = '', path: str = 'runs/') -> None:
    self.stamp = Stamp()
    self.batch_size = batch_size
    self.writer = SummaryWriter(
      log_dir=f'{path}{self.stamp.get_hex()}_{self.stamp.timestamp()}_{comment}',
      comment=f'{self.stamp.get_details()}',
      filename_suffix=f'{self.stamp.timestamp()}'
    )
    self.model_saved = False
    self.tokenizer = tokenizer

  def add_text_to_image(self, image_tensor, captions, outputs, font_size=20):

    # Garantir que as entradas são válidas
    assert image_tensor.dim() == 4, "O batch de imagens deve ter 4 dimensões [B, C, H, W]."
    assert len(captions) == image_tensor.size(0), "O número de legendas deve ser igual ao número de imagens no batch."
    assert len(outputs) == image_tensor.size(0), "O número de predições deve ser igual ao número de imagens no batch."

    processed_images = []

    for i in range(image_tensor.size(0)):
      captions_text = self.tokenizer.decode(captions[i].cpu().numpy(), skip_special_tokens=True)
      predicted_indices = torch.argmax(outputs[i], dim=-1)
      output_text = self.tokenizer.decode(predicted_indices[i].cpu().numpy().tolist(), skip_special_tokens=True)

      # image_tensor = image_tensor[i]  # Seleciona a i-ésima imagem do batch
    
      image_np = image_tensor[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

      # Manipula com PIL
      image_pil = Image.fromarray(image_np)
      draw = ImageDraw.Draw(image_pil)

      # Configura fonte
      try:
          font = ImageFont.truetype("arial.ttf", font_size)
      except IOError:
          font = ImageFont.load_default()

      # Calcula posições dinâmicas
      text_margin = 10
      top_position = text_margin
      bottom_position = image_pil.height - font_size - text_margin

      # Adiciona texto
      draw.text((text_margin, top_position), f"Caption: {captions_text}", font=font, fill="white")
      draw.text((text_margin, bottom_position), f"Prediction: {output_text}", font=font, fill="white")

      # Converte de volta para tensor e adiciona à lista
      processed_images.append(torch.from_numpy(np.array(image_pil)).permute(2, 0, 1))

     # Empilha as imagens processadas em um único tensor [B, C, H, W]
    return torch.stack(processed_images)

  def _log_scalar(self, scalar: float, epoch: int, path: str, mean: bool = True) -> None:
    self.writer.add_scalar(path, np.mean(scalar) if mean else scalar, epoch)
    self.writer.flush()

  def log_scalar_train(self, scalar, epoch, scalar_name='BLEU', mean: bool = True):
    self._log_scalar(scalar=scalar, epoch=epoch, path=f'{scalar_name}/Train', mean=False)

  def log_scalar_val(self, scalar, epoch, scalar_name='BLEU', mean: bool = True):
    self._log_scalar(scalar=scalar, epoch=epoch, path=f'{scalar_name}/Val', mean=False)

  def log_scalar_hiper(self, scalar, epoch, scalar_name='LR'):
    self._log_scalar(scalar=scalar, epoch=epoch, path=f'HIPER/{scalar_name}', mean=False)
  
  def log_image(self, images, epoch, path: str = None):
    img_grid = make_grid(images, nrow=self.batch_size)
    self.writer.add_image(path, img_grid, global_step=epoch)
  
  def log_tensors(self, image, captions, outputs, epoch:int, split:str):
    image = (image.detach().cpu() * 255).to(torch.uint8)

    # for i, (image, captions, output) in enumerate(zip(image, captions, output)):
    image_with_captions = self.add_text_to_image(image, captions, outputs)
    #   # img_cap_out = torch.concat([image, captions_text, output_text], dim=0)
    self.log_image(image_with_captions, path=f'tensors/{split}', epoch=epoch)

  def log_tensors_train(self, image, captions, output, epoch: int):
    self.log_tensors(image, captions, output, epoch, 'train')

  def log_tensors_val(self, image, captions, output, epoch: int):
    self.log_tensors(image, captions, output, epoch, 'val')

  def close(self):
    self.writer.close()

  def log_model(self, model, images_input, forced_log: bool = False):
    if not self.model_saved or forced_log:
      print('Log Model')
      self.writer.add_graph(model, images_input)
      self.model_saved = True
  
  def log_embedding(self, features, class_labels, labels):
    self.writer.add_embedding(features, metadata=class_labels, label_img=labels)

  
    