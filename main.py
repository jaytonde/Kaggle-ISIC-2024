import os
import sys
import wandb
import warnings
import numpy as np
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
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoTokenizer, AutoConfig, DataCollatorWithPadding

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


load_dotenv()

class ISICDataset:
    def __init__(self, image_root_dir, df, transform=None):
        self.image_root_dir = image_root_dir
        self.df             = df
        self.transform      = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_id      = self.df.iloc[idx]['isic_id']
        image_path    = os.path.join(self.image_root_dir, f"{image_id}.jpg")
        pil_image     = Image.open(image_path)
        tensor_image  = self.transform(pil_image)
        tensor_target = torch.tensor(self.df.iloc[idx]['target'], dtype = torch.float)
        
        return {'image':tensor_image, 'label':tensor_target}

class ISICModel(L.LightningModule):
    def __init__(self, config, num_classes: int = 2, pretrained: bool = True):
        super(ISICModel, self).__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        self.model    = timm.create_model(config.MODEL_NAME, pretrained=pretrained)
        if "convnext" in config.MODEL_NAME:
            self.model.head.fc    = nn.Linear(self.model.head.fc.in_features, 1)
        elif "efficientnet" in config.MODEL_NAME:
            self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)
        elif "resnet" in config.MODEL_NAME:
            self.model.fc         = nn.Linear(self.model.fc.in_features, 1)
            
        self.save_hyperparameters() # to save all init parameter's as hyperparameter's
   
    def forward(self, x):
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x     = batch['image']
        y     = batch['label']
        y_hat = self(x)
        loss  = self.loss_fn(y_hat, y)
        self.training_step_outputs.append(y_hat)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x     = batch['image']
        y     = batch['label']
        y_hat = self(x)
        loss  = self.loss_fn(y_hat, y)
        self.validation_step_outputs.append(y_hat)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def loss_fn(self, y_hat, y):
        return nn.BCEWithLogitsLoss()(y_hat, y.unsqueeze(1)) #[[TODO]]
    
    def on_train_epoch_end(self):
        all_preds = torch.stack(self.training_step_outputs)
        self.log("It is the end of last epoch", len(all_preds))
        
    def on_validation_epoch_end(self):
        all_preds = torch.stack(self.validation_step_outputs)
        self.log("It is the end of last validation epoch", len(all_preds))
        
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
        self.train_dataset = ISICDataset("/kaggle/input/isic-2024-challenge/train-image/image", self.train_df, self.train_transform)
        self.val_dataset   = ISICDataset("/kaggle/input/isic-2024-challenge/train-image/image", self.val_df, self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)

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

    print(f"All output folder is push to huggingface repo for experiment : {config.experiment_name}")

def inference(config, trainer, eval_dataset, eval_df, out_dir):

    logits, _, _       = trainer.predict(eval_dataset)
    predictions        = logits.argmax(-1) + 1
    eval_df["pred"]   = predictions

    for i in range(6):
        eval_df[f'pred_{i}'] = logits[:,i]

    file_path          = out_dir + '/' +f"fold_{config.fold}_oof.csv"
    eval_df.to_csv(file_path, index=False)

    print(f"OOF is saved at : {file_path} having shape : {eval_df.shape}")

def main(config):

    start_time = datetime.now()

    print(f"Experiment name : {config.experiment_name} having model : {config.model_id} is started..")

    if config.debug:
        print(f"Debugging mode is on.....")
    elif config.full_fit:
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

    # Initialize DataModule
    data_module = ISICDataModule(
        hdf5_file_path  = config.image_path,
        train_df        = train_df,
        val_df          = eval_df,
        train_transform = get_transform("train"),
        test_transform  = get_transform("test"),
        batch_size      = config.batch_size,
        num_workers     = config.num_workers
    )
    
    data_module.setup()

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
    if not config.full_fit:
        val_results = trainer.test(model, test_dataloaders=DataLoader(
            ISICDataset(config.hdf5_file_path, eval_df, get_transform("test")),
            batch_size  = config.batch_size,
            num_workers = config.num_workers
        ))
    

    if config.full_fit:
        print("No inference for full fit")
    else:
        inference(config, trainer, eval_dataset, eval_df, out_dir)

    push_to_huggingface(config, out_dir)
    
    end_time = datetime.now()
    print(f"Total time taken by experiment {(end_time-start_time)/60} minutes.")
    print(f"This is the end.....")

if __name__ == "__main__":
    config_file_path = sys.argv.pop(1)
    cfg              = OmegaConf.merge(OmegaConf.load(config_file_path), OmegaConf.from_cli())
    main(cfg)
