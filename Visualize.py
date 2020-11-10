import matplotlib.pyplot as plt
import torch
import os
from utils import plot_tensor

def plot_samples(name):
    for image in os.listdir('/users/spadel/mount/point/' + name + '/reconstruction'):
        torch_image = torch.load('/users/spadel/mount/point/' + name + '/reconstruction/' + image,
                                 map_location=torch.device('cpu'))
        plot_tensor(torch_image[0])
        print(image)

plot_samples('10epoch_64f_recloss_0_0001')