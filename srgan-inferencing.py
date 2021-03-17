import os
from srgan import *
import argparse
from pathlib import Path
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("file_path", type=Path)

p = parser.parse_args()
#print(p.file_path, type(p.file_path), p.file_path.exists())

generator = torch.load('srgenerator.pt', map_location=torch.device('cpu')) # torch.load('srgenerator.pt')

hr_size = [256, 256]
lr_size = [64, 64]

hr_transforms = transforms.Compose([transforms.Resize(hr_size), transforms.Lambda(lambda img: np.array(img)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

lr_transforms = transforms.Compose([transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)), transforms.ToPILImage(), transforms.Resize(lr_size, interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor()])

hr_real = hr_transforms(Image.open(p.file_path).convert('RGB'))
lr_real = torch.unsqueeze(lr_transforms(hr_real),0)
hr_fake = generator(lr_real)

save_image(show_tensor_images(lr_real * 2 - 1), 'low_res.jpg')
save_image(show_tensor_images(hr_fake.to(hr_real.dtype)), 'high_res.jpg')
#show_tensor_images(hr_real)