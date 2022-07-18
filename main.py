import os
import wandb

from transformers import AutoFeatureExtractor
from pytorch_lightning.loggers import WandbLogger
from roboflow import Roboflow
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from CocoDetection import CocoDetection
from YoloS import YoloS

wandb.login(key=os.environ.get('WANDB_KEY'))  # "030562801c6f6fd6446cb7a66f9727e7ca5b4a73"

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['labels'] = labels
    return batch


rf = Roboflow(api_key=os.environ.get('ROBOFLOW_KEY'))
project = rf.workspace("joseph-nelson").project("bccd")
dataset = project.version(3).download("coco")

feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-small", size=512, max_size=864)

train_dataset = CocoDetection(img_folder=(dataset.location + '/train'), feature_extractor=feature_extractor)
val_dataset = CocoDetection(img_folder=(dataset.location + '/valid'), feature_extractor=feature_extractor, train=False)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1, shuffle=True, num_workers=8)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=1, num_workers=8)
batch = next(iter(train_dataloader))

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k, v in cats.items()}
num_labels = len(id2label)

# initialize the model
model = YoloS(lr=2.5e-5, weight_decay=1e-4, num_labels=num_labels,
              train_dataloader=train_dataloader, val_dataloader=val_dataloader)

wandb.init(project="test-project")
# wandb_logger = WandbLogger()

wandb_logger = WandbLogger(project='test-project', log_model=True)

# Keep track of the checkpoint with the lowest validation loss
checkpoint_callback = ModelCheckpoint(monitor="validation/loss", mode="min")
trainer = Trainer(max_epochs=50, gradient_clip_val=0.1, accumulate_grad_batches=8,  # gpus=1,
                  log_every_n_steps=5, logger=wandb_logger, callbacks=[checkpoint_callback])
#  checkpoint_callback to log model to W&B at end of training and changed log_every_n_steps=5 to generate better charts
trainer.fit(model)
