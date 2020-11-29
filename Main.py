import torch
from DataLoader import ImageDataset, ImageDataset2, DataLoader
from utils import save_model, load_model, convert_image_np, batch_colour_map, load_images_from_folder, plot_tensor, \
                  save, save_heat_map
from Model import Model, Model2
from config import parse_args, write_hyperparameters
from dotmap import DotMap
import os
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
from transformations import tps_parameters, make_input_tps_param, ThinPlateSpline
import kornia.augmentation as K
import matplotlib.pyplot as plt
# from Tests import ThinPlateSpline1
# import tensorflow as tf


def main2(arg):
    # Get args
    bn = arg.bn
    mode = arg.mode
    name = arg.name
    load_from_ckpt = arg.load_from_ckpt
    lr = arg.lr
    weight_decay = arg.weight_decay
    epochs = arg.epochs
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    arg.device = device

    if mode == 'train':
        # Make new directory
        model_save_dir = name + "/training2"
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            os.makedirs(model_save_dir + '/image')
            os.makedirs(model_save_dir + '/reconstruction')
            os.makedirs(model_save_dir + '/mu')
            os.makedirs(model_save_dir + '/parts')

        # Load Datasets
        train_data = load_images_from_folder()[:100]
        train_dataset = ImageDataset2(train_data, arg)
        test_data = load_images_from_folder()[-1000:]
        test_dataset = ImageDataset2(test_data, arg)

        # Prepare Dataloader & Instances
        train_loader = DataLoader(train_dataset, batch_size=bn, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bn)
        model = Model2(arg).to(device)
        if load_from_ckpt == True:
            model = load_model(model, model_save_dir).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Make Training
        for epoch in range(epochs):
            # Train on Train Set
            model.train()
            model.mode = 'train'
            for step, original in enumerate(train_loader):
                original = original.to(device, dtype=torch.float)
                optimizer.zero_grad()
                # plot_tensor(original[0])
                # plot_tensor(spat[0])
                # plot_tensor(app[0])
                # plot_tensor(original[1])
                # plot_tensor(spat[1])
                # plot_tensor(app[1])
                # print(coord, vec)
                prediction, loss = model(original)
                loss.backward()
                optimizer.step()
                if epoch % 2 == 0 and step == 0:
                    print(f'Epoch: {epoch}, Train Loss: {loss}')

            # Evaluate on Test Set
            model.eval()
            for step, original in enumerate(test_loader):
                with torch.no_grad():
                    original = original.to(device, dtype=torch.float)
                    prediction, loss = model(original)
                    if epoch % 2 == 0 and step == 0:
                        print(f'Epoch: {epoch}, Test Loss: {loss}')

            # Track Progress
            if epoch % 5 == 0:
                model.mode = 'predict'
                image, reconstruction, mu, shape_stream_parts, heat_map = model(original)
                for i in range(len(image)):
                    save_image(image[i], model_save_dir + '/image/' + str(i) + '_' + str(epoch) + '.png')
                    save_image(reconstruction[i], model_save_dir + '/image/' + str(i) + '_' + str(epoch) + '.png')
                    #save_image(mu[i], model_save_dir + '/image/' + str(epoch) + '.png')
                    #save_image(shape_stream_parts[i], model_save_dir + '/image/' + str(epoch) + '.png')

            # Save the current Model
            if epoch % 50 == 0:
                save_model(model, model_save_dir)

    elif arg.mode == 'predict':
        model_save_dir = arg.name + "/prediction"
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)


def main(arg):
    # Get args
    bn = arg.bn
    mode = arg.mode
    name = arg.name
    load_from_ckpt = arg.load_from_ckpt
    lr = arg.lr
    epochs = arg.epochs
    device = torch.device('cuda:' + str(arg.gpu) if torch.cuda.is_available() else 'cpu')
    arg.device = device

    if mode == 'train':
        # Make new directory
        model_save_dir = '../results/' + name
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            os.makedirs(model_save_dir + '/image')
            os.makedirs(model_save_dir + '/reconstruction')
            os.makedirs(model_save_dir + '/mu')
            os.makedirs(model_save_dir + '/parts')
            os.makedirs(model_save_dir + '/heat_map')

        # Save Hyperparameters
        write_hyperparameters(arg.toDict(), model_save_dir)

        # Define Model & Optimizer
        model = Model(arg).to(device)
        if load_from_ckpt == True:
            model = load_model(model, model_save_dir).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Load Datasets and DataLoader
        data = load_images_from_folder()
        train_data = np.array(data[:-1000])
        train_dataset = ImageDataset(train_data)
        test_data = np.array(data[-1000:])
        test_dataset = ImageDataset(test_data)
        train_loader = DataLoader(train_dataset, batch_size=bn, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=bn, num_workers=4)

        # Make Training
        for epoch in range(epochs+1):
            # Train on Train Set
            model.train()
            model.mode = 'train'
            for step, original in enumerate(train_loader):
                original = original.to(device)
                # Make transformations
                tps_param_dic = tps_parameters(original.shape[0], arg.scal, arg.tps_scal, arg.rot_scal,
                                               arg.off_scal, arg.scal_var, arg.augm_scal)
                coord, vector = make_input_tps_param(tps_param_dic)
                coord, vector = coord.to(device), vector.to(device)
                image_spatial_t, _ = ThinPlateSpline(original, coord, vector,
                                                     original.shape[3], device)
                image_appearance_t = K.ColorJitter(arg.brightness, arg.contrast, arg.saturation, arg.hue)(original)
                # Zero out gradients
                optimizer.zero_grad()
                prediction, loss = model(original, image_spatial_t, image_appearance_t, coord, vector)
                loss.backward()
                optimizer.step()
                if step == 0:
                    loss_log = torch.tensor([loss])
                else:
                    loss_log = torch.cat([loss_log, torch.tensor([loss])])
            print(f'Epoch: {epoch}, Train Loss: {torch.mean(loss_log)}')

            # Evaluate on Test Set
            model.eval()
            for step, original in enumerate(test_loader):
                with torch.no_grad():
                    original = original.to(device)
                    tps_param_dic = tps_parameters(original.shape[0], arg.scal, arg.tps_scal, arg.rot_scal, arg.off_scal,
                                                   arg.scal_var, arg.augm_scal)
                    coord, vector = make_input_tps_param(tps_param_dic)
                    coord, vector = coord.to(device), vector.to(device)
                    image_spatial_t, _ = ThinPlateSpline(original, coord, vector,
                                                         original.shape[3], device)
                    image_appearance_t = K.ColorJitter(arg.brightness, arg.contrast, arg.saturation, arg.hue)(original)
                    prediction, loss = model(original, image_spatial_t, image_appearance_t, coord, vector)
                    if step == 0:
                        loss_log = torch.tensor([loss])
                    else:
                        loss_log = torch.cat([loss_log, torch.tensor([loss])])
            print(f'Epoch: {epoch}, Test Loss: {torch.mean(loss)}')

            # Track Progress
            if epoch % 5 == 0:
                model.mode = 'predict'
                image, reconstruction, mu, shape_stream_parts, heat_map = model(original, image_spatial_t,
                                                                      image_appearance_t, coord, vector)
                for i in range(len(image)):
                    if epoch == 0:
                        save_image(image[i], model_save_dir + '/image/' + str(i) + '_' + str(epoch) + '.png')
                    save_image(reconstruction[i], model_save_dir + '/reconstruction/' + str(i) + '_' + str(epoch) + '.png')
                    # save_image(shape_stream_parts[0][0],
                    #            model_save_dir + '/parts/' + str(i) + '_' + str(epoch) + '.png')
                    # save_image(heat_map[0][0],
                    #            model_save_dir + '/heat_map/' + str(i) + '_' + str(epoch) + '.png')
                save_model(model, model_save_dir)

    elif mode == 'predict':
        # Make Directory for Predictions
        model_save_dir = '../results/' + name
        if not os.path.exists(model_save_dir + '/predictions'):
            os.makedirs(model_save_dir + '/predictions')
        # Load Model and Dataset
        model = Model(arg).to(device)
        model = load_model(model, model_save_dir).to(device)
        data = load_images_from_folder(stop=True)
        test_data = np.array(data[-4:])
        test_dataset = ImageDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=bn)
        model.mode = 'predict'
        model.eval()
        # Predict on Dataset
        for step, original in enumerate(test_loader):
            with torch.no_grad():
                original = original.to(device)
                tps_param_dic = tps_parameters(original.shape[0], arg.scal, arg.tps_scal, arg.rot_scal, arg.off_scal,
                                               arg.scal_var, arg.augm_scal)
                coord, vector = make_input_tps_param(tps_param_dic)
                coord, vector = coord.to(device), vector.to(device)
                # tf_coord, tf_vector = tf.convert_to_tensor(coord.cpu().detach().numpy()), tf.convert_to_tensor(vector.cpu().detach().numpy())
                # np_tensor = original.permute(0, 2, 3, 1).cpu().detach().numpy()
                # tf_tensor = tf.convert_to_tensor(np_tensor)
                # image_spatial_t, _ = ThinPlateSpline1(tf_tensor, tf_coord, tf_vector,
                #                                      256, 3)
                # image_spatial_t = torch.tensor(image_spatial_t.numpy()).permute(0, 3, 1, 2).to(device)
                image_spatial_t, _ = ThinPlateSpline(original, coord, vector,
                                                     original.shape[3], device)
                image_appearance_t = K.ColorJitter(arg.brightness, arg.contrast, arg.saturation, arg.hue)(original)
                # image_spatial_t, image_appearance_t = transforms.ToTensor()(image_spatial_t.numpy()), \
                #                                       transforms.ToTensor()(image_appearance_t.numpy())
                image, reconstruction, mu, shape_stream_parts, heat_map = model(original, image_spatial_t,
                                                                                image_appearance_t, coord, vector)
                save_image(image[0], model_save_dir + '/predictions/original.png')
                save_image(reconstruction[0], model_save_dir + '/predictions/reconstruction.png')
                save_image(image_spatial_t[0], model_save_dir + '/predictions/spatial_transform.png')
                save_image(image_appearance_t[0], model_save_dir + '/predictions/appearance_transform.png')
                plot_tensor(original[0])
                plt.show()
                plot_tensor(image_spatial_t[0])
                plt.show()
                plot_tensor(image_appearance_t[0])
                plt.show()
                print(torch.max(image_spatial_t[0][0]), torch.min(image_spatial_t[0][0]))
                print(torch.max(image_spatial_t[0][1]), torch.min(image_spatial_t[0][1]))
                print(torch.max(image_spatial_t[0][2]), torch.min(image_spatial_t[0][2]))
                #print(mu[0][0], mu[0][1], mu[0][2], mu[0][3])
                save_heat_map(heat_map[0], model_save_dir)
                # for i in range(len(image)):
                #     save_image(image[i], model_save_dir + '/predictions/original' + str(i) + '.png')
                #     save_image(reconstruction[i],
                #                model_save_dir + '/predictions/reconstruction' + str(i) + '.png')

if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    main(arg)

