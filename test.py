
from PIL import Image
import requests
import matplotlib.pyplot as plt
import tqdm.auto as tqdm

import torch 
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop, Resize
from diffusion import reverse_transform,get_noisy_image
import numpy as np
def test():
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url,stream=True).raw)
    image_size = 128
    transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),
        Lambda(lambda t: (t*2)-1),
    ])
    x_start = transform(image).unsqueeze(0)
    print(x_start.shape)
    plt.imshow(reverse_transform(x_start.squeeze()))
    
    t = torch.tensor([40])
    plt.imshow(get_noisy_image(x_start,t))

    torch.manual_seed(0)

    def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0]) + with_orig
        fig, axs = plt.subplots(figsize=(15,15), nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            row = [image] + row if with_orig else row
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        if with_orig:
            axs[0, 0].set(title='Original image')
            axs[0, 0].title.set_size(8)
        if row_title is not None:
            for row_idx in range(num_rows):
                axs[row_idx, 0].set(ylabel=row_title[row_idx])

        plt.tight_layout()
        
    plot([get_noisy_image(x_start, torch.tensor([t])) for t in [0, 50, 100, 150, 199]])
    #plt.show()