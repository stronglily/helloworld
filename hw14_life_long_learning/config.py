
class configurations(object):
  def __init__(self):
    self.batch_size = 256
    self.num_epochs = 10000
    self.store_epochs = 250
    self.summary_epochs = 250
    self.learning_rate = 0.0005
    self.load_model = False
    self.store_model_path = "./model"
    self.load_model_path = "./model"
    self.data_path = "./data"
    self.mode = None
    self.lifelong_coeff = 0.5
