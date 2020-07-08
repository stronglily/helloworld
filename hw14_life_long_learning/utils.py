import torch
from hw14_life_long_learning.data import Data, Dataloader
from hw14_life_long_learning.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, optimizer, store_model_path):
  # save model and optimizer
  torch.save(model.state_dict(), '{store_model_path}.ckpt')
  torch.save(optimizer.state_dict(), '{store_model_path}.opt')
  return


def load_model(model, optimizer, load_model_path):
  # load model and optimizer
  print('Load model from {load_model_path}')
  model.load_state_dict(torch.load('{load_model_path}.ckpt'))
  optimizer.load_state_dict(torch.load('{load_model_path}.opt'))
  return model, optimizer


def build_model(data_path, batch_size, learning_rate):
  # create model
  model = Model().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  data = Data(data_path)
  datasets = data.get_datasets()
  tasks = []
  for dataset in datasets:
    tasks.append(Dataloader(dataset, batch_size))

  return model, optimizer, tasks
