import os

import wandb
import torch
import torchvision.utils as vutils
from tqdm import tqdm
from piq import FID, ssim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model_G,
        model_D,
        optimizer_G,
        optimizer_D,
        criterion,
        data_loaders,
        device,
        n_epochs: int,
        len_epoch: int = None,
        log_step: int = None,
        wandb_project: str = 'dcgan',
        wandb_run: str = 'baseline',
        save_dir: str = 'saved',
        save_every: int = 5,
        compute_eval_every: int = 5,
        start_epoch: int = 0,
        **kwargs
    ):
        assert 'train' in data_loaders and 'eval' in data_loaders
        
        self.device = device
        self.model_D = model_D.to(device)
        self.model_G = model_G.to(device)
        
        self.optimizer_D = optimizer_D
        self.optimizer_G = optimizer_G
        
        self.criterion = criterion.to(device)
        
        self.train_loader = data_loaders['train']
        self.valid_loader = data_loaders['eval']
        
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch
        # ----------LOGGING----------
        self.setup_logger(wandb_project, wandb_run)
        self.log_step = log_step
        
        self.train_losses_D = []
        self.train_losses_G = []
        
        self.grad_norms_D = []
        self.grad_norms_G = []
        
        if len_epoch is None:
            self.len_epoch = len(self.train_loader)
        else:
            self.train_loader = inf_loop(self.train_loader)
            self.len_epoch = len_epoch 
        
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.save_every = save_every
        self.compute_eval_every = compute_eval_every
        self.fid = FID()
        
        # -----------CONSTANTS-----------
        self.real_label = 1.0
        self.fake_label = 0.0
        self.fixed_noise = torch.randn(64, self.model_G.latent_dim, 1, 1, device=self.device)
        
    def setup_logger(self, project_name, run_name):
        wandb.login()
        wandb.init(
            project=project_name,
            name=run_name
        )
        self.logger = wandb
    
    def step_D(self, real_batch):
        self.optimizer_D.zero_grad()

        # real batch
        real_images = real_batch.to(self.device)
        B = real_images.shape[0]
        labels = torch.full((B,), self.real_label, dtype=torch.float, device=self.device)
        logits = self.model_D(real_images).view(-1)
        
        err_D_real = self.criterion(logits, labels)
        err_D_real.backward()
        
        # fake batch
        noise = torch.randn(B, self.model_G.latent_dim, 1, 1, device=self.device)
        fake_images = self.model_G(noise)
        labels.fill_(self.fake_label)
        logits = self.model_D(fake_images.detach()).view(-1)
        
        err_D_fake = self.criterion(logits, labels)
        err_D_fake.backward()
        
        self.optimizer_D.step()
        
        return (err_D_real + err_D_fake).detach().cpu().numpy(), fake_images
    
    def step_G(self, fake_batch):
        self.optimizer_G.zero_grad()
        
        fake_images = fake_batch.to(self.device)
        B = fake_images.shape[0]
        labels = torch.full((B,), self.real_label, dtype=torch.float, device=self.device)
        logits = self.model_D(fake_images).view(-1)
        
        err_G = self.criterion(logits, labels)
        err_G.backward()
        
        self.optimizer_G.step()
        
        return err_G.detach().cpu().numpy()
    
    def train_epoch(self, epoch):
        self.model_D.train()
        self.model_G.train()
        
        for batch_idx, batch in enumerate(
            tqdm(self.train_loader, desc=f'Training epoch {epoch + 1}/{self.n_epochs}', 
                 total=self.len_epoch)
        ):
            if batch_idx == self.len_epoch:
                break

            err_D, fake_batch = self.step_D(batch)
            err_G = self.step_G(fake_batch)
            
            self.train_losses_D += [err_D]
            self.train_losses_G += [err_G]
            
            self.grad_norms_D += [self.get_grad_norm(model='D')]
            self.grad_norms_G += [self.get_grad_norm(model='G')]
            
            if batch_idx % self.log_step == 0:
                step = epoch * self.len_epoch + batch_idx
                
                mean_loss_D = sum(self.train_losses_D) / len(self.train_losses_D)
                mean_loss_G = sum(self.train_losses_G) / len(self.train_losses_G)
                
                mean_grad_norm_D = sum(self.grad_norms_D) / len(self.grad_norms_D)
                mean_grad_norm_G = sum(self.grad_norms_G) / len(self.grad_norms_G)
                
                self.logger.log({"train_loss_D": mean_loss_D}, step=step)
                self.logger.log({"train_loss_G": mean_loss_G}, step=step)
                self.logger.log({"grad_norm_D": mean_grad_norm_D}, step=step)
                self.logger.log({"grad_norm_G": mean_grad_norm_G}, step=step)
                
                self.train_losses_D.clear()
                self.train_losses_G.clear()
                self.grad_norms_D.clear()
                self.grad_norms_G.clear()
    
    def eval_epoch(self, epoch):
        self.model_G.eval()
        with torch.inference_mode():
            fake_batch_scaled = self.model_G(self.fixed_noise).detach().cpu()
        
        fake_batch = fake_batch_scaled * 0.5 + 0.5
        eval_batch = None
        for batch in self.valid_loader:
            eval_batch = batch[:64, :, :, :]
            break
        eval_batch = eval_batch * 0.5 + 0.5
        
        ssim_index = ssim(fake_batch, eval_batch, data_range=1.)
        
        fake_loader = DataLoader(fake_batch, collate_fn=lambda x: {"images": torch.stack(x, dim=0)})
        eval_loader = DataLoader(eval_batch, collate_fn=lambda x: {"images": torch.stack(x, dim=0)})
        
        fake_features = self.fid.compute_feats(fake_loader)
        eval_features = self.fid.compute_feats(eval_loader)
        
        fid = self.fid(fake_features, eval_features)
        
        image_grid = vutils.make_grid(fake_batch_scaled, padding=2, normalize=True)
        
        step = (epoch + 1) * self.len_epoch
        self.logger.log({
            "FID": fid,
            "SSIM": ssim_index,
            "generated_images": self.logger.Image(image_grid)
        }, step=step)
    
    def train(self):
        try:
            for epoch in range(self.start_epoch, self.n_epochs):
                self.train_epoch(epoch)
                if epoch % self.compute_eval_every == 0:
                    self.eval_epoch(epoch)
                if epoch % self.save_every == 0:
                    self.save_checkpoint(epoch)
        except KeyboardInterrupt as e:
            print('Saving model on keyboard interrupt...')
            self.save_checkpoint(epoch)
            raise e
        
        self.save_checkpoint(self.n_epochs)

    @torch.no_grad()
    def get_grad_norm(self, model: str = 'G', norm_type=2):
        if model == 'G':
            parameters = self.model_G.parameters()
        elif model == 'D':
            parameters = self.model_D.parameters()
        else:
            raise ValueError()

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        
        return total_norm.detach().cpu().numpy()

    def save_checkpoint(self, epoch):
        state = {
            "epoch": epoch,
            "state_dict_G": self.model_G.state_dict(),
            "state_dict_D": self.model_D.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict()
        }
        filename = os.path.join(self.save_dir, f'checkpoint-epoch{epoch}.pth')
        print('Saving checkpoint...')
        torch.save(state, filename)

    def resume_from_checkpoint(self, ckpt_path, resume_only_model: bool = False):
        state = torch.load(ckpt_path)
        self.model_G.load_state_dict(state['state_dict_G'])
        self.model_D.load_state_dict(state['state_dict_D'])
        if not resume_only_model:
            self.optimizer_G.load_state_dict(state['optimizer_G'])
            self.optimizer_D.load_state_dict(state['optimizer_D'])

        print('State loaded from checkpoint.')
