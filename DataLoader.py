from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import kornia.augmentation as K
from transformations import tps_parameters, make_input_tps_param, ThinPlateSpline


class ImageDataset(Dataset):
    def __init__(self, images, device, bn=1):
        super(ImageDataset, self).__init__()
        self.device = device
        self.bn = bn
        self.images = images
        self.transforms_image = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize([0.5], [0.5])
                                                    ])
        self.transforms_shape = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize([0.5], [0.5])
                                                    ])
        self.transforms_appearance = transforms.Compose([transforms.ToPILImage(),
                                                         transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize([0.5], [0.5])
                                                         ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Select Image
        image = self.images[index]

        # Get parameters for transformations
        tps_param_dic = tps_parameters(self.bn)
        coord, vector = make_input_tps_param(tps_param_dic)

        # Make transformations
        x_spatial_transform = self.transforms_shape(image).unsqueeze(0)
        x_spatial_transform, t_mesh = ThinPlateSpline(x_spatial_transform, coord, vector, 128, device=self.device)
        x_spatial_transform = x_spatial_transform.squeeze(0)
        x_appearance_transform = self.transforms_appearance(image)
        original = self.transforms_image(image)
        coord, vector = coord.squeeze(0), vector.squeeze(0)

        return original, x_spatial_transform, x_appearance_transform, coord, vector
