import torch
import matplotlib.pyplot as plt
from DataLoader import ImageDataset, ImageDataset2, DataLoader
from utils import save_model, load_model, convert_image_np, batch_colour_map, load_images_from_folder, plot_tensor
from Model import Model, Model2
from config import parse_args, write_hyperparameters
from dotmap import DotMap
import os
import numpy as np
from torchvision.utils import save_image


def main2(arg):
    # Get args
    bn = arg.bn
    mode = arg.mode
    name = arg.name
    load_from_ckpt = arg.load_from_ckpt
    lr = arg.lr
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
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

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
                image, reconstruction, mu, shape_stream_parts = model(original)
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

        # Save Hyperparameters
        #write_hyperparameters(arg.toDict(), model_save_dir)

        # Load Datasets
        data = load_images_from_folder()
        train_data = np.array(data[:-1000])
        train_dataset = ImageDataset(train_data, arg)
        test_data = np.array(data[-1000:])
        test_dataset = ImageDataset(test_data, arg)

        # Prepare Dataloader & Instances
        train_loader = DataLoader(train_dataset, batch_size=bn, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bn)
        model = Model(arg).to(device)
        if load_from_ckpt == True:
            model = load_model(model, model_save_dir).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        # Make Training
        for epoch in range(epochs+1):
            # Train on Train Set
            model.train()
            model.mode = 'train'
            for step, (original, spat, app, coord, vec) in enumerate(train_loader):
                original, spat, app, coord, vec = original.to(device, dtype=torch.float), spat.to(device, dtype=torch.float), \
                                                  app.to(device, dtype=torch.float), coord.to(device, dtype=torch.float), \
                                                  vec.to(device, dtype=torch.float)
                optimizer.zero_grad()
                prediction, loss = model(original, spat, app, coord, vec)
                loss.backward()
                optimizer.step()
                if step == 0:
                    loss_log = torch.tensor([loss])
                else:
                    loss_log = torch.cat([loss_log, torch.tensor([loss])])
            print(f'Epoch: {epoch}, Train Loss: {torch.mean(loss_log)}')

            # Evaluate on Test Set
            model.eval()
            for step, (original, spat, app, coord, vec) in enumerate(test_loader):
                with torch.no_grad():
                    original, spat, app, coord, vec = original.to(device, dtype=torch.float), spat.to(device, dtype=torch.float), \
                                                      app.to(device, dtype=torch.float), coord.to(device, dtype=torch.float), \
                                                      vec.to(device, dtype=torch.float)
                    prediction, loss = model(original, spat, app, coord, vec)
                    if step == 0:
                        loss_log = torch.tensor([loss])
                    else:
                        loss_log = torch.cat([loss_log, torch.tensor([loss])])
            print(f'Epoch: {epoch}, Test Loss: {torch.mean(loss)}')

            # Track Progress
            if epoch % 2 == 0:
                model.mode = 'predict'
                image, reconstruction, mu, shape_stream_parts = model(original, spat, app, coord, vec)
                for i in range(len(image)):
                    save_image(image[i], model_save_dir + '/image/' + str(i) + '_' + str(epoch) + '.png')
                    save_image(reconstruction[i], model_save_dir + '/reconstruction/' + str(i) + '_' + str(epoch) + '.png')
                save_model(model, model_save_dir)


if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    main(arg)
    # reconstruction_0 = torch.load(arg.name + '/reconstruction/300.pt')[2]
    # plot_tensor(reconstruction_0)
    # reconstruction_10 = torch.load(arg.name + '/reconstruction/500.pt')[2]
    # plot_tensor(reconstruction_10)
    # reconstruction_20 = torch.load(arg.name + '/reconstruction/800.pt')[2]
    # plot_tensor(reconstruction_20)
    # reconstruction_30 = torch.load(arg.name + '/reconstruction/995.pt')[2]
    # plot_tensor(reconstruction_30)

