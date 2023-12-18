import warnings
from dataclasses import dataclass, asdict

import torch
import numpy as np
import torchvision.utils as vutils

from dcgan import Generator


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


@dataclass
class Config:
    latent_dim: int = 100
    fmap_dim: int = 64
    in_channels: int = 3
    out_channels: int = 3
    
    ckpt_path: str = 'saved/DCGAN/checkpoint-epoch50.pth'


def generate(samples: int = 64, device: torch.device = 'cuda:0'):
    cfg = Config()
    generator = Generator(**asdict(cfg)).to(device)
    generator.load_state_dict(torch.load(cfg.ckpt_path)['state_dict_G'])
    generator.eval()
    noise = torch.randn(samples, cfg.latent_dim, 1, 1, device=device)
    
    with torch.inference_mode():
        generated_images = generator(noise).detach().cpu()
    
    image_grid = vutils.make_grid(generated_images, padding=2, normalize=True)
    vutils.save_image(image_grid, 'grid.png')


if __name__ == '__main__':
    generate()
