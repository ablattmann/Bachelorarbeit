import torch
import torch.nn as nn
from transformations import ThinPlateSpline


def get_local_part_appearances(f, sig):
    alpha = torch.einsum('bfij, bkij -> bkf', f, sig)
    return alpha


def get_mu_and_prec(part_maps, device, scal=0.8):
    """
        Calculate mean for each channel of part_maps
        :param part_maps: tensor of part map activations [bn, n_part, h, w]
        :return: mean calculated on a grid of scale [-1, 1]
        """
    bn, nk, h, w = part_maps.shape
    y_t = torch.linspace(-1., 1., h).reshape(h, 1).repeat(1, w).unsqueeze(-1)
    x_t = torch.linspace(-1., 1., w).reshape(1, w).repeat(h, 1).unsqueeze(-1)
    meshgrid = torch.cat((y_t, x_t), dim=-1).to(device) # 64 x 64 x 2

    mu = torch.einsum('ijl, akij -> akl', meshgrid, part_maps) # 1 x 20 x 2
    mu_out_prod = torch.einsum('akm,akn->akmn', mu, mu)

    mesh_out_prod = torch.einsum('ijm,ijn->ijmn', meshgrid, meshgrid)
    stddev = torch.einsum('ijmn,akij->akmn', mesh_out_prod, part_maps) - mu_out_prod

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


def precision_dist_op(precision, dist, part_depth, nk, h, w):
    proj_precision = torch.einsum('bnik,bnkf->bnif', precision, dist) ** 2  # tf.matmul(precision, dist)**2
    proj_precision = torch.sum(proj_precision, -2)  # sum x and y axis

    heat = 1 / (1 + proj_precision)
    heat = heat.reshape(-1, nk, h, w)  # bn width height number parts
    part_heat = heat[:, :part_depth]
    part_heat = part_heat
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


def feat_mu_to_enc(features, mu, L_inv, device, static=True, n_reverse=1, covariance=None, feat_shape=None,
                   heat_feat_normalize=True, range=10):
    """
    :param features: tensor shape   bn, nk, nf
    :param mu: tensor shape  [bn, nk, 2] in range[-1,1]
    :param L_inv: tensor shape  [bn, nk, 2, 2]
    :param n_reverse:
    :return:
    """
    bn, nk, nf = features.shape
    # for rec_dim = 128
    reconstruct_stages = [[128, 128], [64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
    part_depths = [nk, nk, nk, nk, 4, 2]
    feat_map_depths = [[0, 0], [0, 0], [0, 0], [4, nk], [2, 4], [0, 2]]

    if static:
        reverse_features = torch.cat((features[bn//2:], features[:bn//2]), 0)

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

            heat_feat_map = torch.einsum('bkij,bkn->bnij', heat_scal, feature_slice_rev)


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


def total_loss(input, reconstr, sig_shape, sig_app, coord, vector, device, L_mu=5., L_cov=0.1):
    # Equiv Loss
    channels = sig_shape.shape[2]
    sig_shape_trans, _ = ThinPlateSpline(sig_shape, coord, vector, channels, device=device)
    mu_1, L_inv1 = get_mu_and_prec(sig_app, device)
    cov_1 = torch.einsum('bnij, bnjk -> bnik', L_inv1.transpose(2, 3), L_inv1)
    mu_2, L_inv2 = get_mu_and_prec(sig_shape_trans, device)
    cov_2 = torch.einsum('bnij, bnjk -> bnik', L_inv2.transpose(2, 3), L_inv2)
    equiv_loss = torch.sum(L_mu * torch.norm(mu_1 - mu_2, p=2, dim=2) + \
                           L_cov * torch.norm(cov_1 - cov_2, p=1, dim=[2, 3]), dim=1)

    # Rec Loss
    rec_loss = nn.MSELoss()(input, reconstr)

    total_loss = torch.mean(equiv_loss + rec_loss)
    return total_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



