import io
import os
import sys
import timm
import h5py
import torch
import wandb
import warnings
import numpy as np
from torch import nn
import pandas as pd
import lightning as L
from PIL import Image
from io import BytesIO
from datetime import datetime
from datasets import Dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf
import huggingface_hub as hf_hub
from huggingface_hub import HfApi
from huggingface_hub import login
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoTokenizer, AutoConfig, DataCollatorWithPadding, set_seed

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


load_dotenv()

class ISICDataset:
    def __init__(self, image_file, df, transform=None):
        self.df             = df
        self.transform      = transform
        self.image_file     = h5py.File(image_file, 'r')
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_id      = self.df.iloc[idx]['isic_id']
        image_data    = self.image_file[image_id][()]
        pil_image     = Image.open(io.BytesIO(image_data))
        tensor_image  = self.transform(pil_image)
        tensor_target = torch.tensor(self.df.iloc[idx]['target'], dtype = torch.float)
        
        return {'image':tensor_image, 'label':tensor_target}

class ISICModel(L.LightningModule):
    def __init__(self, config, num_classes: int = 2, pretrained: bool = True):
        super(ISICModel, self).__init__()
        self.validation_step_outputs       = []
        self.validation_step_ground_truths = []
        
        self.model                = timm.create_model(config.model_id, pretrained=pretrained)
        if "convnext" in config.model_id:
            self.model.head.fc    = nn.Linear(self.model.head.fc.in_features, 1)
        elif "efficientnet" in config.model_id:
            self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)
        elif "resnet" in config.model_id:
            self.model.fc         = nn.Linear(self.model.fc.in_features, 1)
   
    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x     = batch['image']
        y     = batch['label']
        y_hat = self(x)
        loss  = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x     = batch['image']
        y     = batch['label']
        y_hat = self(x)
        loss  = self.loss_fn(y_hat, y)
        self.validation_step_outputs.append(y_hat)
        self.validation_step_ground_truths.append(y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def loss_fn(self, y_hat, y):
        return nn.BCEWithLogitsLoss()(y_hat, y.unsqueeze(1)) #[[TODO]]
    
    def on_train_epoch_end(self):
        self.log("It is the end of last epoch")
        
    def on_validation_epoch_end(self):
        all_preds  = torch.cat(self.validation_step_outputs).cpu().numpy()
        all_labels = torch.cat(self.validation_step_ground_truths).cpu().numpy()

        threshold         = 0.5 
        all_labels_binary = (all_labels > threshold).astype(int)
        all_preds_proba   = 1 / (1 + np.exp(-all_preds))

        unique_labels = np.unique(all_labels)
        if len(unique_labels) < 2:
            print(f"Warning: Only one class present in labels. Unique labels: {unique_labels}")
            metric = 0.5  # default value 
        else:
            metric = roc_auc_score(all_labels, all_preds_proba)
        
        self.validation_step_outputs.clear()
        self.validation_step_ground_truths.clear()
        self.log("ROC AUC metric", metric)
        
    def predict_step(self, batch):
        x = batch['image']
        y = batch['label']
        return self(x)
        
class ISICDataModule(L.LightningDataModule):
    def __init__(self, hdf5_file_path, train_df, val_df, train_transform=None, test_transform=None, batch_size=32, num_workers=4):
        super().__init__()
        self.hdf5_file_path   = hdf5_file_path
        self.train_df         = train_df
        self.val_df           = val_df
        self.train_transform  = train_transform
        self.test_transform   = test_transform
        self.batch_size       = batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = ISICDataset(self.hdf5_file_path, self.train_df, self.train_transform)
            self.val_dataset   = ISICDataset(self.hdf5_file_path, self.val_df, self.test_transform)
        elif stage == "test":
            self.test_dataset  = ISICDataset(self.hdf5_file_path, self.val_df, self.test_transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)

def get_transform(mode):
    if mode == "train":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Convert PIL Image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
        ])
        return transform

    elif mode == "test":
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image to tensor
        ])
        return transform
    else:
        print("Wrong transfromation mode selected. it should train/test")

def push_to_huggingface(config, out_dir):
    
    login(token=os.environ["HF_TOKEN"], write_permission=True)  

    repo_id = os.environ["HF_USERNAME"] + '/' + config.experiment_name
    api     = HfApi()
    
    print(f"Uploading files to huggingface repo...")

    if config.full_fit:
        repo_url     = hf_hub.create_repo(repo_id, exist_ok=True, private=True)
        path_in_repo = f"full_fit"
        api.upload_folder(
            folder_path=out_dir, repo_id=repo_id, path_in_repo=path_in_repo
        )
    else:
        repo_url     = hf_hub.create_repo(repo_id, exist_ok=True, private=True)
        path_in_repo = f"fold_{config.fold}"
        api.upload_folder(
            folder_path=out_dir, repo_id=repo_id, path_in_repo=path_in_repo
        )
    
    print(f"Current working dir : {os.getcwd()}")

    api.upload_file(
        path_or_fileobj=config.train_code_file,
        path_in_repo="experiment.py",
        repo_id=repo_id,
        repo_type="model",
        )
    api.upload_file(
        path_or_fileobj="config.yaml",
        path_in_repo="config.yaml",
        repo_id=repo_id,
        repo_type="model",
        )

    print(f"All output folder files are pushed to huggingface repo for experiment : {config.experiment_name}")

def save_results(config, eval_df, results):

    eval_df['preds_thre'] = test_results

    stacked               = torch.stack(tensor_list)
    eval_df["preds"]      = torch.clamp(stacked, min=0.0, max=1.0)


    file_path          = out_dir + '/' +f"fold_{config.fold}_oof.csv"
    eval_df.to_csv(file_path, index=False)

    print(f"OOF is saved at : {file_path} having shape : {eval_df.shape}")

def main(config):

    start_time = datetime.now()

    print(f"Experiment name : {config.experiment_name} having model : {config.model_id} is started..")

    if config.debug:
        print(f"Debugging mode is on.....")
    if config.full_fit:
        print(f"Running experiment in full_fit mode.....")
        out_dir = os.path.join(config.output_dir,f"full_fit")
    else:
        print(f"Running experiment in folding mode.....")
        out_dir = os.path.join(config.output_dir,f"fold_{config.fold}")

    os.makedirs(out_dir, exist_ok = True)

    set_seed(config.seed)

    if config.wandb_log:
        name = ''
        if config.full_fit:
            name = "full_fit"
        else:
            name = f"fold_{config.fold}"

        wandb.init(
                        project = config.wandb_project_name,
                        group   = config.experiment_name,
                        name    = name,
                        notes   = config.notes,
                        config  = OmegaConf.to_container(config, resolve=True)
                )

    dataset_df        = pd.read_csv(os.path.join(config.data_dir,config.training_filename))

    if config.debug:
        train_df          = dataset_df[0:1000]
        eval_df           = dataset_df[1001:2050]
    else:    
        if config.full_fit:
            train_df          = dataset_df
            eval_df           = None
        else:
            train_df          = dataset_df[dataset_df["fold"] != config.fold]
            eval_df           = dataset_df[dataset_df["fold"] == config.fold]

    print(f"Shape of the train df : {train_df.shape}")
    print(f"Shape of the test df : {eval_df.shape}")

    print(f"Unique labels in train_df : {train_df['target'].unique()}")
    print(f"Unique labels in eval_df  : {eval_df['target'].unique()}")

    # Initialize DataModule
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Convert PIL Image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
    ])
    test_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image to tensor
        ])

    print(f"Image path : {config.image_path}")
    data_module = ISICDataModule(
        hdf5_file_path  = config.image_path,
        train_df        = train_df,
        val_df          = eval_df,
        train_transform = train_transform,
        test_transform  = test_transform,
        batch_size      = config.batch_size,
        num_workers     = config.num_workers
    )
    
    data_module.setup(stage="fit")

    # Initialize model
    model = ISICModel(config)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath    = f'checkpoints/fold_{config.fold}',
        filename   = 'best-checkpoint',
        save_top_k = 1,
        verbose    = True,
        monitor    = 'val_loss',
        mode       = 'min'
    )

    # Initialize trainer
    trainer = Trainer(
        max_epochs  = config.max_epochs,
        callbacks   = [checkpoint_callback],
        accelerator = "gpu",
        precision   = "16-mixed",
    )

    # Train the model
    trainer.fit(model, data_module)

    # Evaluate on validation set
    if config.full_fit:
        print("No inference for full fit")

    else:
        test_results = trainer.test(ckpt_path="best", datamodule=data_module)
    
    save_results(config, eval_df, val_results)
    push_to_huggingface(config, out_dir)
    
    end_time = datetime.now()
    print(f"Total time taken by experiment {(end_time-start_time)/60} minutes.")
    print(f"This is the end.....")

if __name__ == "__main__":
    config_file_path = sys.argv.pop(1)
    cfg              = OmegaConf.merge(OmegaConf.load(config_file_path), OmegaConf.from_cli())
    main(cfg)
