class EarlyStopping:
  def __init__(self, patience: int=5, delta:float=0.0) -> None:
    self._patience = patience
    self._delta = delta
    self._counter = 0
    self._best_loss = None
    self._early_stop = False

  def __call__(self, loss):
    if self._best_loss is None:
      self._best_loss = loss

    elif loss > self._best_loss - self._delta:
      self._counter += 1
      if self._counter >= self._patience:
        self._early_stop = True
    else:
      self._best_loss = loss
      self._counter = 0
    
    print(f'Best loss: {(self._best_loss):0.3f} | Current loss:{(loss):0.3f} | strikes: {self._counter}')
    print(' ')
  
  def early_stop(self):
    return self._early_stop