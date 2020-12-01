import torch
import os
from utils import plot_tensor
import numpy as np
import cv2


def plot_samples(name):
    for image in os.listdir('/users/spadel/mount/point/' + name + '/reconstruction'):
        torch_image = torch.load('/users/spadel/mount/point/' + name + '/reconstruction/' + image,
                                 map_location=torch.device('cpu'))
        plot_tensor(torch_image[0])
        print(image)


def make_img_grid(appearance_t, shape_t, pred, tgt, n_logged=4, target_label="Targets",
                  label_app = "Appearance-transformed", label_gen = "Predictions", label_shape = "Shape-transformed"):
    appearance = (appearance_t.permute(0, 2, 3, 1).cpu().numpy() * 255.).astype(np.uint8)[:n_logged]
    shape = (shape_t.permute(0, 2, 3, 1).cpu().numpy() * 255.).astype(np.uint8)[:n_logged]
    pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255.).astype(
        np.uint8)[:n_logged]

    tgt = (tgt.permute(0, 2, 3, 1).cpu().numpy() * 255.).astype(
        np.uint8)[:n_logged]
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
    grid = np.concatenate([appearance, shape, pred, tgt], axis=0)
    return grid