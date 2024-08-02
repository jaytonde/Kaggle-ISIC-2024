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


class TAB_DATA_Preprocessing():

    def feature_engineering(self, df):

        df["lesion_size_ratio"]              = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
        df["lesion_shape_index"]             = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
        df["hue_contrast"]                   = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
        df["luminance_contrast"]             = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
        df["lesion_color_difference"]        = np.sqrt(df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2)
        df["border_complexity"]              = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
        df["color_uniformity"]               = df["tbp_lv_color_std_mean"] / df["tbp_lv_radial_color_std_max"]
        df["3d_position_distance"]           = np.sqrt(df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2) 
        df["perimeter_to_area_ratio"]        = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]
        df["lesion_visibility_score"]        = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]

        
        df["symmetry_border_consistency"]    = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
        df["color_consistency"]              = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]
        
        df["size_age_interaction"]           = df["clin_size_long_diam_mm"] * df["age_approx"]
        df["hue_color_std_interaction"]      = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
        df["lesion_severity_index"]          = (df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df["tbp_lv_eccentricity"]) / 3
        df["shape_complexity_index"]         = df["border_complexity"] + df["lesion_shape_index"]
        df["color_contrast_index"]           = df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df["tbp_lv_deltaLBnorm"]
        df["log_lesion_area"]                = np.log(df["tbp_lv_areaMM2"] + 1)
        df["normalized_lesion_size"]         = df["clin_size_long_diam_mm"] / df["age_approx"]
        df["mean_hue_difference"]            = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
        df["std_dev_contrast"]               = np.sqrt((df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2) / 3)
        df["color_shape_composite_index"]    = (df["tbp_lv_color_std_mean"] + df["tbp_lv_area_perim_ratio"] + df["tbp_lv_symm_2axis"]) / 3
        df["3d_lesion_orientation"]          = np.arctan2(train["tbp_lv_y"], train["tbp_lv_x"])
        df["overall_color_difference"]       = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]) / 3
        df["symmetry_perimeter_interaction"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
        df["comprehensive_lesion_index"]     = (df["tbp_lv_area_perim_ratio"] + df["tbp_lv_eccentricity"] + df["tbp_lv_norm_color"] + df["tbp_lv_symm_2axis"]) / 4

        # 1. Composite shape index
        df['shape_complexity_ratio'] = df['tbp_lv_norm_border'] / df['lesion_shape_index']
        
        # 2. color variability
        df['color_variability'] = df['tbp_lv_color_std_mean'] / df['tbp_lv_stdL']
        
        # 3. Boundary asymmetry
        df['border_asymmetry'] = df['tbp_lv_norm_border'] * (1 - df['tbp_lv_symm_2axis'])
        
        # 4.Relationship between 3D position and size
        df['3d_size_ratio'] = df['3d_position_distance'] / df['clin_size_long_diam_mm']
        
        # 5. Interaction of age and lesion characteristics
        df['age_lesion_interaction'] = df['age_approx'] * df['lesion_severity_index']
        
        # 6. Composite index of color contrast
        df['color_contrast_complexity'] = df['color_contrast_index'] * df['tbp_lv_radial_color_std_max']
        
        # 7. Composite index of shape and color
        df['shape_color_composite'] = df['shape_complexity_index'] * df['color_uniformity']
        
        # 8. 病変の相対的な大きさ
        df['relative_lesion_size'] = df['clin_size_long_diam_mm'] / df['tbp_lv_minorAxisMM']
        
        # 9. 境界の複雑さと色彩の変動性の相互作用
        df['border_color_interaction'] = df['border_complexity'] * df['color_variability']
        
        # 10. Polar coordinate representation of 3D position
        df['3d_radial_distance'] = np.sqrt(df['tbp_lv_x']**2 + df['tbp_lv_y']**2 + df['tbp_lv_z']**2)
        df['3d_polar_angle'] = np.arccos(df['tbp_lv_z'] / df['3d_radial_distance'])
        df['3d_azimuthal_angle'] = np.arctan2(df['tbp_lv_y'], df['tbp_lv_x'])
        
        # 11. 病変の形状の複雑さと大きさの比
        df['shape_size_ratio'] = df['shape_complexity_index'] / df['clin_size_long_diam_mm']
        
        # 12. 色彩の非一様性と境界の複雑さの複合指標
        df['color_border_complexity'] = df['color_uniformity'] * df['border_complexity']
        
        # 13. 病変の可視性と大きさの相互作用
        df['visibility_size_interaction'] = df['lesion_visibility_score'] * np.log(df['clin_size_long_diam_mm'])
        
        # 14. 年齢調整済みの病変の特徴
        df['age_adjusted_lesion_index'] = df['comprehensive_lesion_index'] / np.log(df['age_approx'])
        
        # 15. 色彩コントラストの非線形変換
        df['nonlinear_color_contrast'] = np.tanh(df['color_contrast_index'])
        
        # 16. 病変の形状と位置の複合指標
        df['shape_location_index'] = df['lesion_shape_index'] * df['3d_position_distance']
        
        # 17. 境界の複雑さと非対称性の比率
        df['border_complexity_asymmetry_ratio'] = df['border_complexity'] / (df['tbp_lv_symm_2axis'] + 1e-5)
        
        # 18. 色彩の変動性と病変の大きさの相互作用
        df['color_variability_size_interaction'] = df['color_variability'] * np.log(df['tbp_lv_areaMM2'])
        
        # 19. 3D位置と病変の特徴の複合指標
        df['3d_lesion_composite'] = df['3d_position_distance'] * df['comprehensive_lesion_index']

        # 20. 病変の形状と色彩の非線形複合指標
        df['nonlinear_shape_color_composite'] = np.tanh(df['shape_color_composite'])
        
        new_num_cols = [
            "lesion_size_ratio", "lesion_shape_index", "hue_contrast",
            "luminance_contrast", "lesion_color_difference", "border_complexity",
            "color_uniformity", "3d_position_distance", "perimeter_to_area_ratio",
            "lesion_visibility_score", "symmetry_border_consistency", "color_consistency",

            "size_age_interaction", "hue_color_std_interaction", "lesion_severity_index", 
            "shape_complexity_index", "color_contrast_index", "log_lesion_area",
            "normalized_lesion_size", "mean_hue_difference", "std_dev_contrast",
            "color_shape_composite_index", "3d_lesion_orientation", "overall_color_difference",
            "symmetry_perimeter_interaction", "comprehensive_lesion_index","shape_complexity_ratio",
            "color_variability", "border_asymmetry", "3d_size_ratio", "age_lesion_interaction",
            "color_contrast_complexity", "shape_color_composite", "relative_lesion_size",
            "border_color_interaction", "3d_radial_distance", "3d_polar_angle", "3d_azimuthal_angle",
            "shape_size_ratio", "color_border_complexity", "visibility_size_interaction", "age_adjusted_lesion_index",
            "nonlinear_color_contrast", "shape_location_index", "border_complexity_asymmetry_ratio", "color_variability_size_interaction",
            "3d_lesion_composite", "nonlinear_shape_color_composite",
        ]
    #     new_cat_cols = ["combined_anatomical_site"]
        return df, new_num_cols


class ISIC_HYBRIDDataset:
    def __init__(self, csv, hdf5, mode, meta_features, transform=None):
        self.csv = csv
        if csv is not None and mode != "test":
            self.patient_0   = csv.query(f"target == 0").reset_index(drop=True)
            self.patient_1   = csv.query(f"target == 1").reset_index(drop=True)
        else:
            self.hdf5        = hdf5
            self.patient_ids = list(self.hdf5.keys())

        self.mode          = mode
        self.use_meta      = meta_features is not None
        self.meta_features = meta_features
        self.transform     = transform

    def __len__(self):
        return self.patient_0.shape[0] if self.csv is None else len(self.patient_ids)

    def __getitem__(self, index):

        if self.mode != "test":
            if random.random() > 0.5:
                row = self.patient_1.iloc[index % len(self.patient_1)]
            else:
                row = self.patient_0.iloc[index % len(self.patient_0)]
            image   = cv2.imread(row.image_path)

        else:
            if self.use_meta:
                row    = self.csv.iloc[index]

            image_data = self.hdf5[self.patient_ids[index]][()]
            image      = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res   = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)
    
    def __getitem__(self, idx):
        image_id      = self.df.iloc[idx]['isic_id']
        year          = self.df.iloc[idx]['year']

        if year == 2024:
            image_data    = self.image_file_2024[image_id][()]
            pil_image     = Image.open(io.BytesIO(image_data))
            pil_image     = np.array(pil_image)
        elif year == 2020:
            pil_image    = self.image_file_2020[image_id][()]
        else:
            pil_image    = self.image_file_2019[image_id][()]   
        
        tensor_image  = self.transform(image=pil_image)
        tensor_target = torch.tensor(self.df.iloc[idx]['target'], dtype = torch.float)

        return {'image':tensor_image['image'], 'label':tensor_target}

class ISIC_HYBRIDModel(nn.Module):
    def __init__(self, model_id, out_dim, n_meta_features=0, n_meta_dim=[128,64,32], pretrained=False):
        super(Effnet_Melanoma, self).__init__()

        self.model           = timm.create_model(model_id, pretrained=pretrained, num_classes=0, global_pool='avg')  # Use global pooling
        self.dropouts        = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        in_ch                = self.model.num_features  
        self.n_meta_features = n_meta_features
       
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                nn.SiLU(),
                nn.Dropout(p=0.5),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                nn.SiLU(),
                nn.Dropout(p=0.5),
                nn.Linear(n_meta_dim[1],n_meta_dim[2])
            )
            in_ch += n_meta_dim[2]
        
        self.out     = nn.Linear(in_ch, out_dim)
        self.sigmoid = nn.Sigmoid()
    
    def extract(self, x):
        x = self.model(x)
        return x

    def forward(self, x, x_meta):
        x_img = self.extract(x)  # No need to squeeze as global_pool='avg' already reduces dimensions properly

        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                x_img  = dropout(x_img)
            else:
                x_img += dropout(x_img)
                
        x_img /= len(self.dropouts)
        X      = torch.cat((x_img, x_meta), dim=1)
        out    = self.out(X)
        out    = self.sigmoid(out)
        
        return out

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
    image_file       = h5py.File(config.image_path, 'r')
    eval_df['image'] = eval_df.apply(lambda row: load_image(row, image_file), axis=1) 

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











