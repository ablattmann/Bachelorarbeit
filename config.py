import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="First Try", type=str, help="name of the experiment")

    # run setting
    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--load_from_ckpt', default=False, type=bool)

    # dataset folder
    #parser.add_argument('--dataset', choices=dataset_map_train.keys(), required=True)

    # options
    parser.add_argument('--covariance', default=True, type=bool)
    parser.add_argument('--epochs', default=1000, type=int, help="number of epochs")

    # modes
    parser.add_argument('--bn', default=16, type=int,  help="batchsize if not slim and 2 * batchsize if slim")
    parser.add_argument('--n_parts', default=16, type=int, help="number of parts")
    parser.add_argument('--n_features', default=64, type=int,  help="neurons of feature map layer")
    parser.add_argument('--n_c', default=3, type=int)
    parser.add_argument('--residual_dim', default=256, type=int,  help="neurons in residual module of the hourglass")
    parser.add_argument('--depth_s', default=4, type=int, help="depth of shape hourglass")
    parser.add_argument('--depth_a', default=1, type=int, help="depth of appearance hourglass")

    # loss multiplication constants
    parser.add_argument('--lr',  default=0.0001, type=float, help="learning rate of network")
    parser.add_argument('--L_mu', default=5., type=float, help="")
    parser.add_argument('--L_cov', default=0.1, type=float, help="")

    # tps parameters
    parser.add_argument('--L_inv_scal', default=1., type=float, help="")
    parser.add_argument('--scal', default=1., type=float, nargs='+', help="default 0.6 sensible schedule [0.6, 0.6]")
    parser.add_argument('--tps_scal', default=0.3, type=float, nargs='+', help="sensible schedule [0.01, 0.08]")
    parser.add_argument('--rot_scal', default=0.2, type=float, nargs='+', help="sensible schedule [0.05, 0.6]")
    parser.add_argument('--off_scal', default=0.15, type=float, nargs='+', help="sensible schedule [0.05, 0.15]")
    parser.add_argument('--scal_var', default=0.05, type=float, nargs='+', help="sensible schedule [0.05, 0.2]")
    parser.add_argument('--augm_scal', default=1., type=float, nargs='+', help="sensible schedule [0.0, 1.]")

    #appearance parameters
    parser.add_argument('--brightness_var', default=0.3, type=float,  help="contrast variation")
    parser.add_argument('--contrast_var', default=0.3, type=float, help="contrast variation")
    parser.add_argument('--saturation_var', default=0.1, type=float, help="contrast variation")
    parser.add_argument('--hue_var', default=0.3, type=float,  help="contrast variation")
    parser.add_argument('--p_flip', default=0., type=float, help="contrast variation")
    parser.add_argument('--static', default=True)
    arg = parser.parse_args()
    return arg


def write_hyperparameters(r, save_dir):
    filename = save_dir + "config.txt"
    with open(filename, "a") as input_file:
        for k, v in r.items():
            line = '{}, {}'.format(k, v)
            print(line)
            print(line, file=input_file)