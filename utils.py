import torch
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import os
from skimage.transform import pyramid_reduce


def preprocess_image(image):
    image = image / 255.
    return image


def batch_colour_map(heat_map, device):
    c = heat_map.shape[1]
    colour = []
    for i in range(c):
        colour.append(cm.hsv(float(i / c))[:3])     # does that work?
    colour = torch.tensor(colour, dtype=torch.float).to(device)
    colour_map = torch.einsum('bkij, kl -> blij', heat_map, colour)
    return colour_map


def np_batch_colour_map(heat_map, device):
    c = heat_map.shape[1]
    colour = []
    for i in range(c):
        colour.append(cm.hsv(float(i / c))[:3])
    np_colour = np.array(colour).to(device)
    colour_map = np.einsum('bkij,kl->blij', heat_map, np_colour)
    return colour_map


def identify_parts(image, raw, n_parts, version):
    image_base = np.array(Image.fromarray(image[0]).resize((64, 64))) / 255.
    base = image_base[:, :, 0] + image_base[:, :, 1] + image_base[:, :, 2]
    directory = os.path.join('../images/' + str(version) + "/identify/")
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(n_parts):
        plt.imshow(raw[0, :, :, i] + 0.02 * base, cmap='gray')
        fname = directory + str(i) + '.png'
        plt.savefig(fname, bbox_inches='tight')


def save(img, mu, counter):
    batch_size, out_shape = img.shape[0], img.shape[1:3]
    marker_list = ["o", "v", "s", "|", "_"]
    directory = os.path.join('../images/landmarks/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    s = out_shape[0] // 8
    n_parts = mu.shape[-2]
    mu_img = (mu + 1.) / 2. * np.array(out_shape)[0]
    steps = batch_size
    step_size = 1

    for i in range(0, steps, step_size):
        plt.imshow(img[i])
        for j in range(n_parts):
            plt.scatter(mu_img[i, j, 1], mu_img[i, j, 0],  s=s, marker=marker_list[np.mod(j, len(marker_list))], color=cm.hsv(float(j / n_parts)))

        plt.axis('off')
        fname = directory + str(counter) + '_' + str(i) + '.png'
        plt.savefig(fname, bbox_inches='tight')
        plt.close()


def part_to_color_map(encoding_list, part_depths, size, device, square=True):
    part_maps = encoding_list[0][:, :part_depths[0], :, :]
    if square:
        part_maps = part_maps ** 4
    color_part_map = batch_colour_map(part_maps, device)
    color_part_map = torch.nn.Upsample(size=(size, size))(color_part_map)

    return color_part_map


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.cpu().detach().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    return inp


def save_model(model, model_save_dir):
    torch.save(model.state_dict(), model_save_dir + '/parameters')


def load_model(model, model_save_dir):
    model.load_state_dict(torch.load(model_save_dir + '/parameters'))
    return model


def load_images_from_folder():
    folder = "/export/scratch2/compvis_datasets/deepfashion_vunet/train/"
    images = []
    for i, filename in enumerate(os.listdir(folder)):
        if i == 100:
            break
        img = plt.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(pyramid_reduce(img, downscale=2, multichannel=True))
    return images


def plot_tensor(tensor):
    np_tensor = convert_image_np(tensor)
    plt.imshow(np_tensor)
    plt.show()