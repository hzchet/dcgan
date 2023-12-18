import warnings
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import PixelDataset
from dcgan import Generator, Discriminator
from trainer import Trainer


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


@dataclass
class Config:
    root: str = 'data'
    batch_size: int = 128
    
    latent_dim: int = 100
    fmap_dim: int = 64
    in_channels: int = 3
    out_channels: int = 3
    
    lr: float = 2e-4
    start_epoch: int = 50
    n_epochs: int = 120
    log_step: int = 50
    wandb_project: str = 'pixel generating'
    wandb_run: str = 'DCGAN'
    save_dir: str = 'saved/DCGAN'
    save_every: int = 5
    compute_eval_every: int = 5
    ckpt_path: str = 'saved/DCGAN/checkpoint-epoch50.pth'
    resume_only_model: bool = False


def run():
    cfg = Config()
    
    train_dataset = PixelDataset(cfg.root, 'train')
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=4, pin_memory=True, 
                              shuffle=True, drop_last=True)
    
    valid_dataset = PixelDataset(cfg.root, 'eval')
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, num_workers=4, pin_memory=True, 
                              shuffle=False)
    data_loaders = {"train": train_loader, "eval": valid_loader}
    
    model_D = Discriminator(**asdict(cfg))
    trainable_params_D = filter(lambda p: p.requires_grad, model_D.parameters())
    total_params_D = sum(p.numel() for p in model_D.parameters())
    print(f"Total number of parameters in Discriminator: {total_params_D}")
    
    model_G = Generator(**asdict(cfg))
    trainable_params_G = filter(lambda p: p.requires_grad, model_G.parameters())
    total_params_G = sum(p.numel() for p in model_G.parameters())
    print(f"Total number of parameters in Generator: {total_params_G}")
    
    optimizer_D = torch.optim.Adam(trainable_params_D, cfg.lr)
    optimizer_G = torch.optim.Adam(trainable_params_G, cfg.lr)
    
    criterion = nn.BCELoss()
    
    trainer = Trainer(
        model_G,
        model_D, 
        optimizer_G,
        optimizer_D, 
        criterion, 
        data_loaders,
        device='cuda:0', 
        **asdict(cfg)
    )
    if hasattr(cfg, 'ckpt_path'):
        trainer.resume_from_checkpoint(cfg.ckpt_path, resume_only_model=cfg.resume_only_model)
    
    trainer.train()


if __name__ == '__main__':
    run()
