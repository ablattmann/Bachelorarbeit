import torch
import torch.nn as nn
from torchvision.models import vgg19
import kornia.augmentation as K
import torch.nn.functional as F
from opt_einsum import contract
from collections import namedtuple

from software.transformations import ThinPlateSpline
from software.utils import get_member
from software.architecture_ops import softmax

def get_local_part_appearances(f, sig):
    alpha = contract('bfij, bkij -> bkf', f, sig)
    return alpha


def get_covariance(tensor):
    bn, nk, w, h = tensor.shape
    tensor_reshape = tensor.reshape(bn, nk, 2, -1)
    x = tensor_reshape[:, :, 0, :]
    y = tensor_reshape[:, :, 1, :]
    mean_x = torch.mean(x, dim=2).unsqueeze(-1)
    mean_y = torch.mean(y, dim=2).unsqueeze(-1)

    xx = torch.sum((x - mean_x) * (x - mean_x), dim=2).unsqueeze(-1) / (h * w / 2 - 1)
    xy = torch.sum((x - mean_x) * (y - mean_y), dim=2).unsqueeze(-1) / (h * w / 2 - 1)
    yx = xy
    yy = torch.sum((y - mean_y) * (y - mean_y), dim=2).unsqueeze(-1) / (h * w / 2 - 1)

    cov = torch.cat((xx, xy, yx, yy), dim=2)
    cov = cov.reshape(bn, nk, 2, 2)
    return cov


def get_mu_and_prec(part_maps, device, scal):
    """
        Calculate mean for each channel of part_maps
        :param part_maps: tensor of part map activations [bn, n_part, h, w]
        :return: mean calculated on a grid of scale [-1, 1]
        """
    bn, nk, h, w = part_maps.shape
    y_t = torch.linspace(-1., 1., h).reshape(h, 1).repeat(1, w).unsqueeze(-1)
    x_t = torch.linspace(-1., 1., w).reshape(1, w).repeat(h, 1).unsqueeze(-1)
    meshgrid = torch.cat((y_t, x_t), dim=-1).to(device) # 64 x 64 x 2

    mu = contract('akij,ijl-> akl', part_maps, meshgrid) # 1 x 20 x 2
    mu_out_prod = contract('akm,akn->akmn', mu, mu)

    mesh_out_prod = contract('ijm,ijn->ijmn', meshgrid, meshgrid)
    stddev = contract('akij,ijmn->akmn', part_maps, mesh_out_prod) - mu_out_prod

    a_sq = stddev[:, :, 0, 0]
    a_b = stddev[:, :, 0, 1]
    b_sq_add_c_sq = stddev[:, :, 1, 1]
    eps = 1e-12

    a = torch.sqrt(a_sq + eps)  # Σ = L L^T Prec = Σ^-1  = L^T^-1 * L^-1  ->looking for L^-1 but first L = [[a, 0], [b, c]
    b = a_b / (a + eps)
    c = torch.sqrt(b_sq_add_c_sq - b ** 2 + eps)
    z = torch.zeros_like(a)

    det = (a * c).unsqueeze(-1).unsqueeze(-1)
    row_1 = torch.cat((c.unsqueeze(-1), z.unsqueeze(-1)), dim=-1).unsqueeze(-2)
    row_2 = torch.cat((-b.unsqueeze(-1), a.unsqueeze(-1)), dim=-1).unsqueeze(-2)

    L_inv = scal / (det + eps) * torch.cat((row_1, row_2), dim=-2)  # L^⁻1 = 1/(ac)* [[c, 0], [-b, a]
    return mu, L_inv


def get_heat_map(mu, L_inv, device):
    h, w, nk = 64, 64, L_inv.shape[1]

    y_t = torch.linspace(-1., 1., h).reshape(h, 1).repeat(1, w).unsqueeze(-1)
    x_t = torch.linspace(-1., 1., w).reshape(1, w).repeat(h, 1).unsqueeze(-1)

    y_t_flat = y_t.reshape(1, 1, 1, -1)
    x_t_flat = x_t.reshape(1, 1, 1, -1)

    mesh = torch.cat((y_t_flat, x_t_flat), dim=-2).to(device)
    dist = mesh - mu.unsqueeze(-1)
    proj_precision = contract('bnik, bnkf -> bnif', L_inv, dist) ** 2  # tf.matmul(precision, dist)**2
    proj_precision = torch.sum(proj_precision, -2)  # sum x and y axis
    heat = 1 / (1 + proj_precision)
    heat = heat.reshape(-1, nk, h, w)  # bn number parts width height

    return heat


def precision_dist_op(precision, dist, part_depth, nk, h, w):
    proj_precision = contract('bnik, bnkf -> bnif', precision, dist) ** 2  # tf.matmul(precision, dist)**2
    proj_precision = torch.sum(proj_precision, -2)  # sum x and y axis
    heat = 1 / (1 + proj_precision)
    heat = heat.reshape(-1, nk, h, w)  # bn number parts width height
    part_heat = heat[:, :part_depth]
    return heat, part_heat


def reverse_batch(tensor, n_reverse):
    """
    reverses order of elements the first axis of tensor
    example: reverse_batch(tensor=tf([[1],[2],[3],[4],[5],[6]), n_reverse=3) returns tf([[3],[2],[1],[6],[5],[4]]) for n reverse 3
    :param tensor:
    :param n_reverse:
    :return:
    """
    bn, rest = tensor.shape[0], tensor.shape[1:]
    assert ((bn / n_reverse).is_integer())
    tensor = torch.reshape(tensor, shape=[bn // n_reverse, n_reverse, *rest])
    tensor_rev = tensor.flip(dims=[1])
    tensor_rev = torch.reshape(tensor_rev, shape=[bn, *rest])
    return tensor_rev


def feat_mu_to_enc(features, mu, L_inv, device, covariance, reconstr_dim, static=True, n_reverse=2, feat_shape=True,
                   heat_feat_normalize=True, range=10):
    """
    :param features: tensor shape   bn, nk, nf
    :param mu: tensor shape  [bn, nk, 2] in range[-1,1]
    :param L_inv: tensor shape  [bn, nk, 2, 2]
    :param n_reverse:
    :return:
    """
    bn, nk, nf = features.shape
    if reconstr_dim == 128:
        reconstruct_stages = [[128, 128], [64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
        feat_map_depths = [[0, 0], [0, 0], [0, 0], [4, nk], [2, 4], [0, 2]]
        part_depths = [nk, nk, nk, nk, 4, 2]
    elif reconstr_dim == 256:
        reconstruct_stages = [[256, 256], [128, 128], [64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
        feat_map_depths = [[0, 0], [0, 0], [0, 0], [0, 0], [4, nf], [2, 4], [0, 2]]
        part_depths = [nk, nk, nk, nk, nk, 4, 2]

    if static:
        #reverse_features = torch.cat([features[bn // 2:], features[:bn // 2]], dim=0)
        reverse_features = features
    else:
        reverse_features = reverse_batch(features, n_reverse)

    encoding_list = []
    circular_precision = range * torch.eye(2).reshape(1, 1, 2, 2).to(dtype=torch.float).repeat(bn, nk, 1, 1).to(device)


    for dims, part_depth, feat_slice in zip(reconstruct_stages, part_depths, feat_map_depths):
        h, w = dims[0], dims[1]

        y_t = torch.linspace(-1., 1., h).reshape(h, 1).repeat(1, w).unsqueeze(-1)
        x_t = torch.linspace(-1., 1., w).reshape(1, w).repeat(h, 1).unsqueeze(-1)

        y_t_flat = y_t.reshape(1, 1, 1, -1)
        x_t_flat = x_t.reshape(1, 1, 1, -1)

        mesh = torch.cat((y_t_flat, x_t_flat), dim=-2).to(device)
        dist = mesh - mu.unsqueeze(-1)

        if not covariance or not feat_shape:
            heat_circ, part_heat_circ = precision_dist_op(circular_precision, dist, part_depth, nk, h, w)

        if covariance or feat_shape:
            heat_shape, part_heat_shape = precision_dist_op(L_inv, dist, part_depth, nk, h, w)

        nkf = feat_slice[1] - feat_slice[0]

        if nkf != 0:
            feature_slice_rev = reverse_features[:, feat_slice[0]: feat_slice[1]]

            if feat_shape:
                heat_scal = heat_shape[:, feat_slice[0]: feat_slice[1]]

            else:
                heat_scal = heat_circ[:, feat_slice[0]: feat_slice[1]]

            if heat_feat_normalize:
                heat_scal_norm = torch.sum(heat_scal, 1, keepdim=True) + 1
                heat_scal = heat_scal / heat_scal_norm

            heat_feat_map = contract('bkij,bkn->bnij', heat_scal, feature_slice_rev)


            if covariance:
                encoding_list.append(torch.cat((part_heat_shape, heat_feat_map), 1))

            else:
                encoding_list.append(torch.cat((part_heat_circ, heat_feat_map), 1))

        else:
            if covariance:
                encoding_list.append(part_heat_shape)

            else:
                encoding_list.append(part_heat_circ)

    return encoding_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def AbsDetJacobian(batch_meshgrid, device):
    """
        :param device:
        :param batch_meshgrid: takes meshgrid tensor of dim [bn, 2, h, w] (conceptually meshgrid represents a two dimensional function f = [fx, fy] on [bn, h, w] )
        :return: returns Abs det of  Jacobian of f of dim [bn, 1, h, w]
        """
    y_c = batch_meshgrid[:, 0, :, :].unsqueeze(1)
    x_c = batch_meshgrid[:, 1, :, :].unsqueeze(1)
    sobel_x_filter = 1 / 4 * torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float).reshape(1, 1, 3, 3).cuda(device)
    sobel_y_filter = sobel_x_filter.permute(0, 1, 3, 2).cuda(device)

    filtered_y_y = F.conv2d(y_c, sobel_y_filter, stride=1, padding=1)
    filtered_y_x = F.conv2d(y_c, sobel_x_filter, stride=1, padding=1)
    filtered_x_y = F.conv2d(x_c, sobel_y_filter, stride=1, padding=1)
    filtered_x_x = F.conv2d(x_c, sobel_x_filter, stride=1, padding=1)

    Det = torch.abs(filtered_y_y * filtered_x_x - filtered_y_x * filtered_x_y)

    return Det


def augm(t, arg):
    t = K.ColorJitter(arg.brightness, arg.contrast, arg.saturation, arg.hue)(t)
    random_tensor = 1. - arg.p_flip + torch.rand(size=[1], dtype=t.dtype)
    binary_tensor = torch.floor(random_tensor)
    random_tensor, binary_tensor = random_tensor.to(arg.gpu), binary_tensor.to(arg.gpu)

    augmented = binary_tensor * t + (1 - binary_tensor) * (1 - t)
    return augmented


def prepare_pairs(t_images, train, static, config):
    if train:
        bn, n_c, w, h = t_images.shape
        t_c_1_images = augm(t_images, config)
        t_c_2_images = augm(t_images, config)

        if static:
            t_c_1_images = torch.cat([t_c_1_images[:bn // 2].unsqueeze(1), t_c_1_images[bn // 2:].unsqueeze(1)], dim=1)
            t_c_2_images = torch.cat([t_c_2_images[:bn // 2].unsqueeze(1), t_c_2_images[bn // 2:].unsqueeze(1)], dim=1)
        else:
            t_c_1_images = t_c_1_images.reshape(bn // 2, 2, n_c, h, w)
            t_c_2_images = t_c_2_images.reshape(bn // 2, 2, n_c, h, w)

        a, b = t_c_1_images[:, 0].unsqueeze(1), t_c_1_images[:, 1].unsqueeze(1)
        c, d = t_c_2_images[:, 0].unsqueeze(1), t_c_2_images[:, 1].unsqueeze(1)

        if static:
            t_input_images = torch.cat([a, d], dim=0).reshape(bn, n_c, w, h)
            t_reconstr_images = torch.cat([c, b], dim=0).reshape(bn, n_c, w, h)
        else:
            t_input_images = torch.cat([a, d], dim=1).reshape(bn, n_c, w, h)
            t_reconstr_images = torch.cat([c, b], dim=1).reshape(bn, n_c, w, h)

        t_input_images = torch.clamp(t_input_images, min=0., max=1.)
        t_reconstr_images = F.interpolate(torch.clamp(t_reconstr_images, min=0., max=1.), size=config.reconstr_dim)

    else:
        t_input_images = torch.clamp(t_images, min=0., max=1.)
        t_reconstr_images = F.interpolate(torch.clamp(t_images, min=0., max=1.), size=config.reconstr_dim)

    return t_input_images, t_reconstr_images


def heat_map_function(y_dist, x_dist, y_scale, x_scale):
    x = 1 / (1 + (torch.square(y_dist / (1e-6 + y_scale)) + torch.square(x_dist / (1e-6 + x_scale))))
    return x


def fold_img_with_mu(img, mu, scale, threshold, device, normalize=True):
    """
        folds the pixel values of img with potentials centered around the part means (mu)
        :param img: batch of images
        :param mu:  batch of part means in range [-1, 1]
        :param scale: scale that governs the range of the potential
        :param visualize:
        :param normalize: whether to normalize the potentials
        :return: folded image
        """
    bn, nc, h, w = img.shape
    bn, nk, _ = mu.shape

    py = mu[:, :, 0].unsqueeze(2)
    px = mu[:, :, 1].unsqueeze(2)

    y_t = torch.linspace(-1., 1., h).reshape(h, 1).repeat(1, w)
    x_t = torch.linspace(-1., 1., w).reshape(1, w).repeat(h, 1)
    x_t_flat = x_t.reshape(1, 1, -1).to(device)
    y_t_flat = y_t.reshape(1, 1, -1).to(device)

    eps = 1e-6
    y_dist = py - y_t_flat + eps
    x_dist = px - x_t_flat + eps

    heat_scal = heat_map_function(y_dist=y_dist, x_dist=x_dist, x_scale=scale, y_scale=scale)
    heat_scal = heat_scal.reshape(bn, nk, h, w)  # bn width height number parts
    heat_scal = contract("bkij -> bij", heat_scal)
    heat_scal = torch.clamp(heat_scal, min=0., max=1.)
    heat_scal = torch.where(heat_scal > threshold, heat_scal, torch.zeros_like(heat_scal))

    norm = torch.sum(heat_scal.reshape(bn, -1), dim=1).unsqueeze(1).unsqueeze(1)
    if normalize:
        heat_scal_norm = heat_scal / norm
        folded_img = contract('bcij,bij->bcij', img, heat_scal_norm)
    if not normalize:
        folded_img = contract('bcij,bij->bcij', img, heat_scal)

    return folded_img, heat_scal.unsqueeze(-1)


def fold_img_with_L_inv(img, mu, L_inv, scale, threshold, device, normalize=True):
    """
        folds the pixel values of img with potentials centered around the part means (mu)
        :param img: batch of images
        :param mu:  batch of part means in range [-1, 1]
        :param scale: scale that governs the range of the potential
        :param visualize:
        :param normalize: whether to normalize the potentials
        :return: folded image
        """
    bn, nc, h, w = img.shape
    bn, nk, _ = mu.shape

    mu_stop = mu.detach()

    y_t = torch.linspace(-1., 1., h).reshape(h, 1).repeat(1, w)
    x_t = torch.linspace(-1., 1., w).reshape(1, w).repeat(h, 1)
    x_t_flat = x_t.reshape(1, 1, -1).to(device)
    y_t_flat = y_t.reshape(1, 1, -1).to(device)

    mesh = torch.cat([y_t_flat, x_t_flat], dim=-2)
    eps = 1e-6
    dist = mesh - mu_stop.unsqueeze(-1) + eps

    proj_precision = contract('bnik, bnkf -> bnif', scale * L_inv, dist) ** 2  # tf.matmul(precision, dist)**2
    proj_precision = torch.sum(proj_precision, -2)  # sum x and y axis

    heat = 1 / (1 + proj_precision)

    heat = torch.reshape(heat, shape=[bn, nk, h, w])  # bn width height number parts
    heat = contract('bkij -> bij', heat)
    heat_scal = torch.clamp(heat, min=0., max=1.)
    heat_scal = torch.where(heat_scal > threshold, heat_scal, torch.zeros_like(heat_scal))

    norm = torch.sum(heat_scal.reshape(bn, -1), dim=1).unsqueeze(1).unsqueeze(1)
    if normalize:
        heat_scal = heat_scal / norm
    folded_img = contract('bcij, bij -> bcij', img, heat_scal)

    return folded_img


def loss_fn(bn, mu, L_inv, mu_t, stddev_t, reconstruct_same_id, image_rec, fold_with_shape, l_2_scal, l_2_threshold,
            L_mu, L_cov, device):

    # Equiv Loss
    mu_t_1, mu_t_2 = mu_t[:bn], mu_t[bn:]
    stddev_t_1, stddev_t_2 = stddev_t[:bn], stddev_t[bn:]
    transform_loss = torch.mean((mu_t_1 - mu_t_2) ** 2)

    precision_sq = (stddev_t_1 - stddev_t_2) ** 2

    eps = 1e-6
    precision_loss = torch.mean(torch.sqrt(torch.sum(precision_sq, dim=[2, 3]) + eps))

    img_difference = reconstruct_same_id - image_rec
    distance_metric = torch.abs(img_difference)

    if fold_with_shape:
        fold_img_squared = fold_img_with_L_inv(distance_metric, mu.detach(), L_inv.detach(),
                                               l_2_scal, l_2_threshold, device)
    else:
        fold_img_squared, heat_mask_l2 = fold_img_with_mu(distance_metric, mu, l_2_scal, l_2_threshold, device)

    rec_loss = torch.mean(torch.sum(fold_img_squared, dim=[2, 3]))

    total_loss = rec_loss + L_mu * transform_loss + L_cov * precision_loss
    return total_loss, rec_loss, transform_loss, precision_loss






class PerceptualVGG(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.vgg =  vgg19(pretrained=True)
        self.vgg.eval()

        self.vgg_layers = self.vgg.features


        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float)
            .unsqueeze(dim=0)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1),
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float)
            .unsqueeze(dim=0)
            .unsqueeze(dim=-1)
            .unsqueeze(dim=-1),
        )
        self.target_layers = {
            "3": "relu1_2",
            "8": "relu2_2",
            "13": "relu3_2",
            "15" : "relu3_3",
            "22": "relu4_2",
            "24" : "relu4_3",
            "31": "relu5_2",
        }

        if weights is None:
            self.loss_weights = {"input":1., "relu1_2": 1.,"relu2_2": 1.,"relu3_2": 1.,"relu3_3": 1.,"relu4_2": 1.,"relu4_3": 1.,"relu5_2": 1. }
        else:
            assert isinstance(weights, dict) and list(weights.keys()) == list(self.target_layers.keys()), f"The weights passed to PerceptualVGG have to be a dict with the keys {list(self.target_layers.keys())}"
            self.loss_weights = weights

    def forward(self, x):
        # IMPORTANT: Input is assumed to be in range [0,1] here.
        #x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std


        # add also common reconstruction loss in pixel space
        out = {"input": x}

        for name, submodule in self.vgg_layers._modules.items():
            # x = submodule(x)
            if name in self.target_layers:
                x = submodule(x)
                out[self.target_layers[name]] = x
            else:
                x = submodule(x)

        return out


VGGOutput = namedtuple(
    "VGGOutput",
    ["input", "relu1_2", "relu2_2", "relu3_2", "relu4_2", "relu5_2"],
)

def vgg_loss(custom_vgg:PerceptualVGG, target, pred, weights=None):
    """
    Implements a vgg based perceptual loss, as extensively used for image/video generation tasks
    :param custom_vgg: The vgg feature extractor for the perceptual loss, definition see above
    :param target:
    :param pred:
    :return:
    """
    target_feats = custom_vgg(target)
    pred_feats = custom_vgg(pred)
    target_feats = VGGOutput(**{key: target_feats[key] for key in VGGOutput._fields})
    pred_feats = VGGOutput(**{key: pred_feats[key] for key in VGGOutput._fields})

    names = list(pred_feats._asdict().keys())
    if weights is None:
        losses = {}

        for i, (tf, pf) in enumerate(zip(target_feats, pred_feats)):
            loss = get_member(custom_vgg,"loss_weights")[VGGOutput._fields[i]] * torch.mean(
                torch.abs(tf - pf)
            ).unsqueeze(dim=-1)
            losses.update({names[i]: loss})
    else:

        losses = {
            names[0]: get_member(custom_vgg,"loss_weights")[VGGOutput._fields[0]]
            * torch.mean(weights * torch.abs(target_feats[0] - pred_feats[0]))
            .unsqueeze(dim=-1)
            .to(torch.float)
        }

        for i, (tf, pf) in enumerate(zip(target_feats[1:], pred_feats[1:])):
            loss = get_member(custom_vgg,"loss_weights")[i + 1] * torch.mean(
                torch.abs(tf - pf)
            ).unsqueeze(dim=-1)

            losses.update({names[i + 1]: loss})

    out_loss = torch.stack([losses[key] for key in losses],dim=0,).mean()
    return out_loss
