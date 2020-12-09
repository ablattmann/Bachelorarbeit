import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import colors

from software.ops import get_mu_and_prec,get_heat_map
from software.architecture_ops import softmax


# def make_part_plot(image,part_maps, device):
#     color_list = ['black', 'gray', 'brown', 'chocolate', 'orange', 'gold', 'olive', 'lawngreen', 'aquamarine',
#                   'dodgerblue', 'midnightblue', 'mediumpurple', 'indigo', 'magenta', 'pink', 'springgreen']
#
#     fmap_app_norm = softmax(part_maps)
#     mu, L_inv = get_mu_and_prec(fmap_app_norm, device, scal=5.)
#     heat_map = get_heat_map(mu, L_inv, device)

def get_color_mapping(n_parts):
    colors = []
    for s in np.linspace(0,255,num=n_parts):

        c1 = int(s)
        c2 = 255 - int(s)
        c3 = 127 - int(s) if 127 - int(s) > 0 else 127 + int(s / 2)

        colors.append((c1,c2,c3))

    return colors




def make_visualization(original, reconstruction, shape_transform, app_transform, fmap_shape,
                       fmap_app, directory, epoch, device, index=0):

    # Color List for Parts
    color_list = ['black', 'gray', 'brown', 'chocolate', 'orange', 'gold', 'olive', 'lawngreen', 'aquamarine',
                  'dodgerblue', 'midnightblue', 'mediumpurple', 'indigo', 'magenta', 'pink', 'springgreen']
    # Get Maps
    fmap_shape_norm = softmax(fmap_shape)
    mu_shape, L_inv_shape = get_mu_and_prec(fmap_shape_norm, device, scal=5.)
    heat_map_shape = get_heat_map(mu_shape, L_inv_shape, device)

    fmap_app_norm = softmax(fmap_app)
    mu_app, L_inv_app = get_mu_and_prec(fmap_app_norm, device, scal=5.)
    heat_map_app = get_heat_map(mu_app, L_inv_app, device)


    # Make Head with Overview
    fig_head, axs_head = plt.subplots(3, 4, figsize=(12, 12))
    fig_head.suptitle("Overview", fontsize="x-large")
    axs_head[0, 0].imshow(original[index].permute(1, 2, 0).cpu().detach().numpy())
    axs_head[0, 1].imshow(app_transform[index].permute(1, 2, 0).cpu().detach().numpy())
    axs_head[0, 2].imshow(shape_transform[index].permute(1, 2, 0).cpu().detach().numpy())
    axs_head[0, 3].imshow(reconstruction[index].permute(1, 2, 0).cpu().detach().numpy())

    axs_head[1, 0].imshow(app_transform[index].permute(1, 2, 0).cpu().detach().numpy())
    axs_head[1, 2].imshow(shape_transform[index].permute(1, 2, 0).cpu().detach().numpy())

    axs_head[2, 0].imshow(reconstruction[0].permute(1, 2, 0).cpu().detach().numpy())
    axs_head[2, 1].imshow(reconstruction[1].permute(1, 2, 0).cpu().detach().numpy())
    axs_head[2, 2].imshow(reconstruction[2].permute(1, 2, 0).cpu().detach().numpy())
    axs_head[2, 3].imshow(reconstruction[3].permute(1, 2, 0).cpu().detach().numpy())

    # Part Visualization Shape Stream
    fig_shape, axs_shape = plt.subplots(8, 6, figsize=(8, 8))
    fig_shape.suptitle("Part Visualization Shape Stream", fontsize="x-large")
    for i in range(16):
        cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                        ['white', color_list[i]],
                                                        256)
        if i == 0:
            overlay_shape = heat_map_shape[index][i]
        else:
            overlay_shape += heat_map_shape[index][i]

        axs_shape[int(i / 2), (i % 2) * 3].imshow(fmap_shape[index][i].cpu().detach().numpy(), cmap=cmap)
        axs_shape[int(i / 2), (i % 2) * 3 + 1].imshow(fmap_shape_norm[index][i].cpu().detach().numpy(), cmap=cmap)
        axs_shape[int(i / 2), (i % 2) * 3 + 2].imshow(heat_map_shape[index][i].cpu().detach().numpy(), cmap=cmap)

        if i == 15:
            cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                            ['white', 'black'],
                                                            256)
            axs_head[1, 1].imshow(overlay_shape.cpu().detach().numpy(), cmap=cmap)

    # Part Visualization Appearance Stream
    fig_app, axs_app = plt.subplots(8, 6, figsize=(8, 8))
    fig_app.suptitle("Part Visualization Appearance Stream", fontsize="x-large")
    for i in range(16):
        cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                        ['white', color_list[i]],
                                                        256)
        if i == 0:
            overlay_app = heat_map_app[index][i]
        else:
            overlay_app += heat_map_app[index][i]

        axs_app[int(i / 2), (i % 2) * 3].imshow(fmap_app[index][i].cpu().detach().numpy(), cmap=cmap)
        axs_app[int(i / 2), (i % 2) * 3 + 1].imshow(fmap_app_norm[index][i].cpu().detach().numpy(), cmap=cmap)
        axs_app[int(i / 2), (i % 2) * 3 + 2].imshow(heat_map_app[index][i].cpu().detach().numpy(), cmap=cmap)

        if i == 15:
            cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                            ['white', 'black'],
                                                            256)
            axs_head[1, 3].imshow(overlay_app.cpu().detach().numpy(), cmap=cmap)



    plt.close('all')


def make_keypoint_image(tgts,mus,img_size):
    mus = ((mus + 1) * img_size / 2).cpu().numpy()
    colors = get_color_mapping(mus.shape[1])

    img_list =  []
    for img, kps in zip(tgts,mus):

        for kp, c in zip(kps,colors):
            img = cv2.UMat.get(cv2.circle(cv2.UMat(img),(int(kp[0]),int(kp[1])),int(img_size / 64),c,-1))

        img_list.append(img)

    keypoint_imgs = np.concatenate(img_list,axis=1)

    return cv2.UMat.get(cv2.putText(cv2.UMat(keypoint_imgs), "Keypoint Visualizations", (int(keypoint_imgs.shape[1] // 3), keypoint_imgs.shape[0] - int(keypoint_imgs.shape[0]/6)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    float(keypoint_imgs.shape[0] / 256), (255, 0, 0), int(keypoint_imgs.shape[0] / 128)))


def make_img_grid(appearance_t, shape_t, pred, tgt, mus=None, n_logged=4, target_label="Targets",
                  label_app = "Appearance-transformed", label_gen = "Predictions", label_shape = "Shape-transformed"):
    appearance = (appearance_t.permute(0, 2, 3, 1).cpu().numpy() * 255.).astype(np.uint8)[:n_logged]
    shape = (shape_t.permute(0, 2, 3, 1).cpu().numpy() * 255.).astype(np.uint8)[:n_logged]
    pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255.).astype(
        np.uint8)[:n_logged]

    tgt = (tgt.permute(0, 2, 3, 1).cpu().numpy() * 255.).astype(
        np.uint8)[:n_logged]
    img_size = tgt.shape[1]

    if mus is None:
        keypoint_img = None
    else:
        keypoint_img = make_keypoint_image(tgt, mus[:n_logged],img_size)

    tgt = np.concatenate([t for t in tgt], axis=1)
    tgt = cv2.UMat.get(cv2.putText(cv2.UMat(tgt), target_label , (int(tgt.shape[1] // 3), tgt.shape[0] - int(tgt.shape[0]/6)), cv2.FONT_HERSHEY_SIMPLEX,
                                              float(tgt.shape[0] / 256), (255, 0, 0), int(tgt.shape[0] / 128)))

    appearance = np.concatenate([s for s in appearance], axis=1)
    appearance = cv2.UMat.get(cv2.putText(cv2.UMat(appearance),label_app, (int(appearance.shape[1] // 3), appearance.shape[0] - int(appearance.shape[0]/6)), cv2.FONT_HERSHEY_SIMPLEX,
                                                  float(appearance.shape[0] / 256), (255, 0, 0), int(appearance.shape[0] / 128)))
    shape = np.concatenate([f for f in shape], axis=1)
    shape = cv2.UMat.get(cv2.putText(cv2.UMat(shape), label_shape, (int(shape.shape[1] // 3), shape.shape[0] - int(shape.shape[0]/6)), cv2.FONT_HERSHEY_SIMPLEX,
                                                  float(shape.shape[0] / 256), (255, 0, 0), int(shape.shape[0] / 128)))
    pred = np.concatenate([p for p in pred], axis=1)
    pred = cv2.UMat.get(cv2.putText(cv2.UMat(pred), label_gen, (int(pred.shape[1] // 3), pred.shape[0] - int(pred.shape[0]/6)), cv2.FONT_HERSHEY_SIMPLEX,
                                                  float(pred.shape[0] / 256), (255, 0, 0), int(pred.shape[0] / 128)))

    if keypoint_img is None:
        grid = np.concatenate([appearance, shape, pred, tgt], axis=0)
    else:
        grid = np.concatenate([appearance, shape, keypoint_img, pred, tgt], axis=0)
    return grid