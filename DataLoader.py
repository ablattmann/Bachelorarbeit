from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import kornia.augmentation as K
from transformations import tps_parameters, make_input_tps_param, ThinPlateSpline


class ImageDataset(Dataset):
    def __init__(self, images, arg):
        super(ImageDataset, self).__init__()
        self.device = arg.device
        self.bn = arg.bn
        self.brightness = arg.brightness_var
        self.contrast = arg.contrast_var
        self.saturation = arg.saturation_var
        self.hue = arg.hue_var
        self.scal = arg.scal
        self.tps_scal = arg.tps_scal
        self.rot_scal = arg.rot_scal
        self.off_scal = arg.off_scal
        self.scal_var = arg.scal_var
        self.augm_scal = arg.augm_scal
        self.images = images
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize([0.5], [0.5])
                                              ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Select Image
        image = self.images[index]

        # Get parameters for transformations
        tps_param_dic = tps_parameters(1, self.scal, self.tps_scal, self.rot_scal, self.off_scal,
                                       self.scal_var, self.augm_scal)
        coord, vector = make_input_tps_param(tps_param_dic)

        # Make transformations
        x_spatial_transform = self.transforms(image).unsqueeze(0)
        x_spatial_transform, t_mesh = ThinPlateSpline(x_spatial_transform, coord,
                                                      vector, 128, device='cpu')
        x_spatial_transform = x_spatial_transform.squeeze(0)
        x_appearance_transform = K.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)\
                                (self.transforms(image).unsqueeze(0)).squeeze(0)
        original = self.transforms(image)
        coord, vector = coord[0], vector[0]

        return original, x_spatial_transform, x_appearance_transform, coord, vector


class ImageDataset2(Dataset):
    def __init__(self, images, arg):
        super(ImageDataset2, self).__init__()
        self.images = images
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize([0.5], [0.5])
                                              ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Select Image
        image = self.images[index]
        original = self.transforms(image)

        return original
