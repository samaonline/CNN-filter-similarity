from __future__ import print_function

import copy
import os.path as osp

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from pdb import set_trace as st
import torchvision.datasets as datasets
from tqdm import *

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)


def remove_val(arr):
    arr = arr[arr != 1]
    arr = arr[~np.isnan(arr)]
    return arr

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images


def get_classtable():
    classes = []
    with open("samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@click.command()
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-p", "--data_path", type=str, required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-m", "--model_path", type=str,  required=True, default=None)
@click.option("-o", "--output_dir", type=str, default="./output/res34")
@click.option("--cuda/--cpu", default=True)
def main(data_path, arch, topk, output_dir, model_path, cuda, num_layers=4, VISUALIZE=False):
    """
    Visualize model responses given multiple images
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)
    model = model.to(device)
    #model = torch.nn.DataParallel(model).cuda()#.to(device)
    model.eval()
    
    checkpoint = torch.load(model_path) 
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        model.load_state_dict(checkpoint)
    # Images
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(data_path, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=True,
        num_workers=2, pin_memory=True)


    # =========================================================================
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")
    
    for layer in range(num_layers):
        gbp = GuidedBackPropagation(model=model, layer=5+layer)
        layer_list = []
        for index, (images, target) in tqdm(enumerate(val_loader)):
            images = images.to(device)
            feat = gbp.forward(images)
            grad_list = []
            for i in range(feat.shape[1]):#range(topk):
                # Guided Backpropagation
                gbp.backward(ids=i)
                gradients = gbp.generate()

                grad_list.append(gradients.detach().cpu().view(gradients.shape[0],-1))

                if VISUALIZE:
                    for j in range(len(images)):

                        # Guided Backpropagation
                        save_gradient(
                            filename=osp.join(
                                output_dir,
                                "{}-{}-Sguided-{}-{}.png".format(j, arch,layer, i ),
                            ),
                            gradient=gradients[j],
                        )
            grad_list = torch.stack(grad_list).squeeze().permute(1,0,2)
            for i in grad_list:
                corr_mat = np.abs(np.corrcoef(i.numpy()))
                layer_list.append(remove_val(corr_mat.flatten()))
        np.save(str(layer)+'_18', np.hstack(layer_list) )

if __name__ == "__main__":
    main()
