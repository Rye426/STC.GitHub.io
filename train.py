from share import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu')) 
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

dataset = MyDataset() 
dataloader = DataLoader(dataset, num_workers=55, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)  
trainer = pl.Trainer(devices=1, accelerator="gpu", precision=32, callbacks=[logger],max_epochs=1000)

trainer.fit(model, dataloader)
