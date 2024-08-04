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
import albumentations
from datetime import datetime
from datasets import Dataset
from dotenv import load_dotenv
import torch.nn.functional as F
from omegaconf import OmegaConf
import huggingface_hub as hf_hub
from huggingface_hub import HfApi
from huggingface_hub import login
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, auc, roc_curve
from sklearn.model_selection import KFold
import torchvision.transforms as transforms
from torchmetrics.classification import BinaryAUROC
from sklearn.model_selection import StratifiedKFold
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer, AutoConfig, DataCollatorWithPadding, set_seed
from torchmetrics.classification import BinaryAUROC,BinaryAccuracy,BinaryF1Score
load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")




class ISICDataset:
    def __init__(self, config, df, transform=None):
        self.df                  = df
        self.config              = config
        self.transform           = transform
        self.image_file_2024     = h5py.File(config.image_file_2024, 'r')

        if self.config.use_old_data:
            self.image_file_2020     = h5py.File(config.image_file_2020, 'r')
            self.image_file_2019     = h5py.File(config.image_file_2019, 'r')
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_id      = self.df.iloc[idx]['isic_id']

        if config.use_old_data:
            year          = self.df.iloc[idx]['year']
            if year == 2024:
                image_data    = self.image_file_2024[image_id][()]
                pil_image     = Image.open(io.BytesIO(image_data))
                pil_image     = np.array(pil_image)
            elif year == 2020:
                pil_image    = self.image_file_2020[image_id][()]
            else:
                pil_image    = self.image_file_2019[image_id][()]  
        else:
            image_data    = self.image_file_2024[image_id][()]
            pil_image     = Image.open(io.BytesIO(image_data))
            pil_image     = np.array(pil_image)
    
        tensor_image  = self.transform(image=pil_image)
        tensor_target = torch.tensor(self.df.iloc[idx]['target'], dtype = torch.float)
        tensor_image  = torch.cat([tensor_image['image'], self.F_rgb2hsv(tensor_image['image'])],1)

        return {'image':tensor_image, 'label':tensor_target}

    def F_rgb2hsv(self, rgb: torch.Tensor) -> torch.Tensor:
        cmax, cmax_idx       = torch.max(rgb, dim=1, keepdim=True)
        cmin                 = torch.min(rgb, dim=1, keepdim=True)[0]
        delta                = cmax - cmin
        hsv_h                = torch.empty_like(rgb[:, 0:1, :, :])
        cmax_idx[delta == 0] = 3
        hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsv_h[cmax_idx == 3] = 0.
        hsv_h               /= 6.
        hsv_s                = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
        hsv_v                = cmax
        return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p   = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

class ISICModel(L.LightningModule):

    def __init__(self, config, num_classes: int = 2, pretrained: bool = True):
        super(ISICModel, self).__init__()
        self.config                        = config
        self.train_step_outputs            = []
        self.train_step_ground_truths      = []
        self.validation_step_outputs       = []
        self.validation_step_ground_truths = []
        self.predict_step_outputs          = []

        self.accuracy                      = BinaryAccuracy()
        self.auc_roc                       = BinaryAUROC()
        self.f1_score                      = BinaryF1Score()
        
        self.model                         = timm.create_model(config.model_id, pretrained=pretrained)
        self.pooling                       = GeM()

        if "convnext" in config.model_id:
            self.linear    = nn.Linear(320, 1)
        elif "efficientnet" in config.model_id:
            self.in_features       = self.model.classifier.in_features
            self.model.classifier  = nn.Identity()
            self.model.global_pool = nn.Identity()
            self.linear            = nn.Linear(self.in_features, 1)
        elif "resnet" in config.model_id:
            self.linear    = nn.Linear(self.model.fc.in_features, 1)
            self.dropout   = nn.ModuleList([
                                                nn.Dropout(0.5) for i in range(5)
                                          ])

        if "mobilenet" in config.model_id:
            self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)

        self.save_hyperparameters()
   
    def forward(self, x):
        # logits          = self.model(x)
        # pooled_features = self.pooling(logits).flatten(1)
        # output          = self.linear(pooled_features)

        #convnext 
        # logits          = self.model(x)
        # output          = self.linear(logits)

        #resnet
        logits          = self.model(x)
        pool            = F.adaptive_avg_pool2d(logits,1).reshape(config.batch_size,-1)

        if self.training:
            new_logit = 0
            for i in range(len(self.dropout)):
                new_logit += self.linear(self.dropout[i](pool))
            new_logit = new_logit/len(self.dropout)
        else:
            new_logit = self.linear(pool)

        return new_logit
    
    def training_step(self, batch, batch_idx):
        x      = batch['image']
        y      = batch['label']
        logits = self(x)
        loss   = self.loss_fn(logits, y)
        y_hat  = logits.sigmoid()

        self.train_step_outputs.append(y_hat)
        self.train_step_ground_truths.append(y)
        self.log("train_loss", loss, on_step=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x      = batch['image']
        y      = batch['label']
        logits = self(x)
        loss   = self.loss_fn(logits, y)
        y_hat  = logits.sigmoid()

        self.validation_step_outputs.append(y_hat)
        self.validation_step_ground_truths.append(y)
        self.log("val_loss", loss, on_step=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x      = batch['image']
        y      = batch['label']
        logits = self(x)
        y_hat  = logits.sigmoid()
        self.predict_step_outputs.append(y_hat)
        return y_hat

    def on_train_epoch_end(self):
        all_preds  = torch.cat(self.train_step_outputs)
        all_labels = torch.cat(self.train_step_ground_truths)

        accuracy   = self.accuracy(all_preds,all_labels.unsqueeze(1))
        auc_roc    = self.auc_roc(all_preds,all_labels)
        f1_score   = self.f1_score(all_preds,all_labels.unsqueeze(1))

        self.log_dict({"train_accuracy":accuracy, "train_auc_roc":auc_roc, "train_f1_score":f1_score},
                     on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.train_step_outputs.clear()
        self.train_step_ground_truths.clear()

    def on_validation_epoch_end(self): 
        all_preds      = torch.cat(self.validation_step_outputs).cpu()
        all_labels     = torch.cat(self.validation_step_ground_truths).cpu()

        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    
        mask = tpr >= self.config.tpr_threshold
        if np.sum(mask) < 2:
            pauc = 0.123456
            print("Not enough points above the TPR threshold for pAUC calculation.")
        
        else:
            fpr_above_threshold = fpr[mask]
            tpr_above_threshold = tpr[mask]
            
            partial_auc = auc(fpr_above_threshold, tpr_above_threshold)
            
            pauc = partial_auc * (1 - self.config.tpr_threshold)

        self.log_dict({"pauc":pauc}, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.validation_step_outputs.clear()
        self.validation_step_ground_truths.clear()
        
    def on_predict_epoch_end(self):
        all_preds = torch.cat(self.predict_step_outputs)
        self.predict_step_outputs.clear()
        return all_preds
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer
    
    def loss_fn(self, y_logits, y):
        return nn.BCEWithLogitsLoss()(y_logits, y.unsqueeze(1))
    
class ISICDataModule(L.LightningDataModule):

    def __init__(self, config, train_df, val_df, train_transform=None, test_transform=None, batch_size=32, num_workers=4):
        super().__init__()
        self.config           = config
        self.train_df         = train_df
        self.val_df           = val_df
        self.train_transform  = train_transform
        self.test_transform   = test_transform
        self.batch_size       = batch_size
        self.num_workers      = num_workers

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = ISICDataset(self.config, self.train_df, self.train_transform)
            self.val_dataset   = ISICDataset(self.config, self.val_df, self.test_transform)
        elif stage == "predict":
            self.predict_dataset  = ISICDataset(self.config, self.val_df, self.test_transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

def get_transform(mode, image_size=224):
    if mode == "train":
        transform = albumentations.Compose([
            albumentations.Transpose(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.OneOf([
                albumentations.MotionBlur(blur_limit=5),
                albumentations.MedianBlur(blur_limit=5),
                albumentations.GaussianBlur(blur_limit=5),
                albumentations.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),

            albumentations.OneOf([
                albumentations.OpticalDistortion(distort_limit=1.0),
                albumentations.GridDistortion(num_steps=5, distort_limit=1.),
                albumentations.ElasticTransform(alpha=3),
            ], p=0.7),

            albumentations.CLAHE(clip_limit=4.0, p=0.7),
            albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(),
            ToTensorV2(),
        ])
        return transform

    elif mode == "test":
        transform = albumentations.Compose([
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(),
            ToTensorV2(),
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
        path_in_repo="main.py",
        repo_id=repo_id,
        repo_type="model",
        )
    api.upload_file(
        path_or_fileobj=config.config_file,
        path_in_repo="config.yaml",
        repo_id=repo_id,
        repo_type="model",
        )

    print(f"All output folder files are pushed to huggingface repo for experiment : {config.experiment_name}")

def load_image(row, image_file):
    image_id   = row['isic_id']
    image_data = image_file[image_id][()]
    pil_image  = Image.open(io.BytesIO(image_data))
    return wandb.Image(np.array(pil_image))

def save_results(config, eval_df, results, out_dir, wandb_logger):
    print(f"Length of results : {len(results)}")

    preds            = torch.cat(results)
    eval_df['preds'] = preds

    print("Logging images to wandb.")
    image_file_2024 = h5py.File(config.image_file_2024, 'r')
    image_file_2020 = h5py.File(config.image_file_2020, 'r')
    image_file_2019 = h5py.File(config.image_file_2019, 'r')

    for id, row in eval_df.iterrows():
        if row['year'] == 2024:
            eval_df.at[id,'image'] = load_image(row, image_file_2024)
        elif row['year'] == 2020:
            image_id               = row['isic_id']
            eval_df.at[id,'image'] = wandb.Image(image_file_2020[image_id][()])
        else:
            image_id               = row['isic_id']
            eval_df.at[id,'image'] = wandb.Image(image_file_2019[image_id][()])  

    columns = ['isic_id','image','target','preds']
    data    = eval_df[['isic_id','image','target','preds']].values.tolist()
    wandb_logger.log_table(key="validation data", columns=columns, data=data)

    del eval_df['image']
    print("Logging images to wandb completed successfully.")

    print(f"Shape of the eval_df : {eval_df.shape}")

    file_path             = out_dir + '/' +f"fold_{config.fold}_oof.csv"
    eval_df.to_csv(file_path, index=False)
    print(f"OOF is saved at : {file_path} having shape : {eval_df.shape}")

def main(config):

    start_time = datetime.now()
    print("----------------------------------------------")
    print(os.getenv("KAGGLE_USERNAME"))
    print(os.getenv("WANDB_API_KEY"))    
    print("----------------------------------------------")

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
        wandb_logger = WandbLogger(
                                    project = config.wandb_project_name,
                                    name    = name,
                                    group   = config.experiment_name,
                                    notes   = config.notes,
                                    config  = OmegaConf.to_container(config, resolve=True)
                                    )

    dataset_df            = pd.read_csv(os.path.join(config.data_dir,config.training_filename))

    if config.use_old_data:
        print(f"Shape of the dataset df before up sampling 7 times : {dataset_df.shape}")
        df_2024_mal        = dataset_df[(dataset_df['year']==2024) & (dataset_df['target']==1)]
        df2_duplicated     = pd.concat([df_2024_mal] * 10, ignore_index=True)
        dataset_df         = pd.concat([dataset_df, df2_duplicated], ignore_index=True)
        print(f"Shape of the dataset df after up sampling 7 times : {dataset_df.shape}")

    if config.debug:
        config.max_epochs = 1
        train_df          = dataset_df[0:1000]
        eval_df           = dataset_df[1001:2050]
    else:    
        if config.full_fit:
            train_df          = dataset_df
            eval_df           = None
        else:
            train_df          = dataset_df[(dataset_df["fold"] != config.fold) & (dataset_df["fold"] != -1)]
            eval_df           = dataset_df[dataset_df["fold"] == config.fold]

    print(f"Shape of the train df : {train_df.shape}")
    print(f"Shape of the test df : {eval_df.shape}")

    # Initialize DataModule
    data_module = ISICDataModule(
        config          = config,
        train_df        = train_df,
        val_df          = eval_df,
        train_transform = get_transform("train", config.image_size),
        test_transform  = get_transform("test", config.image_size),
        batch_size      = config.batch_size,
        num_workers     = config.num_workers
    )
    
    data_module.setup(stage="fit")

    # Initialize model
    model = ISICModel(config)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath    = out_dir,
        filename   = 'best-checkpoint',
        save_top_k = 1,
        save_last  = True,
        verbose    = True,
        monitor    = 'val_loss',
        mode       = 'min'
    )

    # Initialize trainer
    trainer = Trainer(
        logger            = wandb_logger,
        log_every_n_steps = 10,
        max_epochs  = config.max_epochs,
        callbacks   = [checkpoint_callback],
        accelerator = "gpu",
        precision   = "16-mixed",
        num_sanity_val_steps = 0,
    )

    # Train the model
    trainer.fit(model, data_module)

    # Evaluate on validation set
    if config.full_fit:
        print("No inference for full fit")

    else:
        test_results = trainer.predict(ckpt_path="best", datamodule=data_module)
    
    save_results(config, eval_df, test_results, out_dir, wandb_logger)
    push_to_huggingface(config, out_dir)
    
    end_time = datetime.now()
    print(f"Total time taken by experiment {(end_time-start_time)/60} minutes.")
    print(f"This is the end.....")

if __name__ == "__main__":
    config_file_path = sys.argv.pop(1)
    cfg              = OmegaConf.merge(OmegaConf.load(config_file_path), OmegaConf.from_cli())
    main(cfg)
