import torch
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import os
from opt_einsum import contract
from torchvision.utils import save_image
import inspect
import coloredlogs
import logging
import logging.config
import yaml


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.cpu().detach().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    return inp


def plot_tensor(tensor):
    np_tensor = convert_image_np(tensor)
    plt.imshow(np_tensor)
    plt.show()


def batch_colour_map(heat_map, device):
    c = heat_map.shape[1]
    colour = []
    for i in range(c):
        colour.append(cm.hsv(float(i / c))[:3])     # does that work?
    colour = torch.tensor(colour, dtype=torch.float).to(device)
    colour_map = contract('bkij, kl -> blij', heat_map, colour)
    return colour_map


def save_heat_map(heat_map, directory):
    for i, part in enumerate(heat_map):
        part = part.unsqueeze(0)
        save_image(part, directory + '/predictions/colourmap_' + str(i) + '.png')


def np_batch_colour_map(heat_map, device):
    c = heat_map.shape[1]
    colour = []
    for i in range(c):
        colour.append(cm.hsv(float(i / c))[:3])
    np_colour = np.array(colour).to(device)
    colour_map = contract('bkij,kl->blij', heat_map, np_colour)
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


def save(img, mu, counter, model_save_dir):
    batch_size, out_shape = img.shape[0], img.shape[1:3]
    marker_list = ["o", "v", "s", "|", "_"]
    directory = os.path.join(model_save_dir + '/predictions/landmarks/')
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


def save_model(model, model_save_dir):
    torch.save(model.state_dict(), model_save_dir + '/parameters')


def load_model(model, model_save_dir):
    model.load_state_dict(torch.load(model_save_dir + '/parameters'))
    return model


def load_images_from_folder(stop=False):
    folder = "/export/scratch2/compvis_datasets/deepfashion_vunet/train/"
    images = []
    for i, filename in enumerate(os.listdir(folder)):
        img = plt.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
        if stop == True:
            if i == 3:
                break
    return images

iuhihfie_logger_loaded = False
def get_logger(name):
    # setup logging
    global iuhihfie_logger_loaded
    if not iuhihfie_logger_loaded:
        with open(f'{os.path.dirname(os.path.abspath(__file__))}/logging.yaml', 'r') as f:
            log_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            logging.config.dictConfig(log_cfg)
            iuhihfie_logger_loaded = True
    logger = logging.getLogger(name)
    coloredlogs.install(logger=logger, level="DEBUG")
    return logger

class LoggingParent:
    def __init__(self):
        super(LoggingParent, self).__init__()
        # find project root
        mypath = inspect.getfile(self.__class__)
        mypath = "/".join(mypath.split("/")[:-1])
        found = False
        while mypath!="" and not found:
            f = []
            for (dirpath, dirnames, filenames) in os.walk(mypath):
                f.extend(filenames)
                break
            if ".gitignore" in f:
                found = True
                continue
            mypath = "/".join(mypath.split("/")[:-1])
        project_root = mypath+"/"
        # Put it together
        file = inspect.getfile(self.__class__).replace(project_root, "").replace("/", ".").split(".py")[0]
        cls = str(self.__class__)[8:-2]
        cls = str(cls).replace("__main__.", "").split(".")[-1]
        self.logger = get_logger(f"{file}.{cls}")