import torch
import matplotlib.pyplot as plt
from DataLoader import ImageDataset, DataLoader
from utils import save_model, load_model, convert_image_np, batch_colour_map, load_images_from_folder, plot_tensor
from Model import Model
from config import parse_args
from dotmap import DotMap
import os
import numpy as np

def main(arg):
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
        model_save_dir = name + "/training"
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            os.makedirs(model_save_dir + '/image')
            os.makedirs(model_save_dir + '/reconstruction')
            os.makedirs(model_save_dir + '/mu')
            os.makedirs(model_save_dir + '/parts')

        # Load Datasets
        train_data = load_images_from_folder()[:30000]
        train_dataset = ImageDataset(train_data, arg)
        test_data = load_images_from_folder()[-1000:]
        test_dataset = ImageDataset(test_data, arg)

        # Prepare Dataloader & Instances
        train_loader = DataLoader(train_dataset, batch_size=bn, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bn)
        model = Model(arg).to(device)
        if load_from_ckpt == True:
            model = load_model(model, model_save_dir).to(device)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

        # Make Training
        for epoch in range(epochs):
            # Train on Train Set
            model.train()
            model.mode = 'train'
            for step, (original, spat, app, coord, vec) in enumerate(train_loader):
                original, spat, app, coord, vec = original.to(device, dtype=torch.float), spat.to(device, dtype=torch.float), \
                                                  app.to(device, dtype=torch.float), coord.to(device, dtype=torch.float), \
                                                  vec.to(device, dtype=torch.float)
                optimizer.zero_grad()
                # plot_tensor(original[0])
                # plot_tensor(spat[0])
                # plot_tensor(app[0])
                # plot_tensor(original[1])
                # plot_tensor(spat[1])
                # plot_tensor(app[1])
                # print(coord, vec)
                prediction, loss = model(original, spat, app, coord, vec)
                loss.backward()
                optimizer.step()
                if epoch % 2 == 0 and step == 0:
                    print(f'Epoch: {epoch}, Train Loss: {loss}')

            # Evaluate on Test Set
            model.eval()
            for step, (original, spat, app, coord, vec) in enumerate(test_loader):
                with torch.no_grad():
                    original, spat, app, coord, vec = original.to(device, dtype=torch.float), spat.to(device, dtype=torch.float), \
                                                      app.to(device, dtype=torch.float), coord.to(device, dtype=torch.float), \
                                                      vec.to(device, dtype=torch.float)
                    prediction, loss = model(original, spat, app, coord, vec)
                    if epoch % 2 == 0 and step == 0:
                        print(f'Epoch: {epoch}, Test Loss: {loss}')

            # Track Progress
            if epoch % 5 == 0:
                model.mode = 'predict'
                image, reconstruction, mu, shape_stream_parts = model(original, spat, app, coord, vec)
                torch.save(image, model_save_dir + '/image/' + str(epoch) + '.pt')
                torch.save(reconstruction, model_save_dir + '/reconstruction/' + str(epoch) + '.pt')
                torch.save(mu, model_save_dir + '/mu/' + str(epoch) + '.pt')
                torch.save(shape_stream_parts, model_save_dir + '/parts/' + str(epoch) + '.pt')

                # Plot
                plot_tensor(image[2])
                plot_tensor(reconstruction[2])
                plot_tensor(shape_stream_parts[2][0].unsqueeze(0))
                plot_tensor(shape_stream_parts[2][1].unsqueeze(0))
                plot_tensor(shape_stream_parts[2][2].unsqueeze(0))

        # Save the current Model
        save_model(model, model_save_dir)

    elif arg.mode == 'predict':
        model_save_dir = arg.name + "/prediction"
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)


if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    main(arg)
    # reconstruction_0 = torch.load('First Try/training/parts/0.pt')[0][0].unsqueeze(0)
    # plot_tensor(reconstruction_0)
    # reconstruction_10 = torch.load('First Try/training/parts/10.pt')[0][0].unsqueeze(0)
    # plot_tensor(reconstruction_10)
    # reconstruction_20 = torch.load('First Try/training/parts/20.pt')[0][0].unsqueeze(0)
    # plot_tensor(reconstruction_20)
    # reconstruction_30 = torch.load('First Try/training/parts/30.pt')[0][0].unsqueeze(0)
    # plot_tensor(reconstruction_30)

