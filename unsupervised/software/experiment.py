import torch
import os
from os import path
from glob import glob
import numpy as np
from torchvision import transforms as T
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Average
import kornia.augmentation as K
import wandb

from software.transformations import tps_parameters, make_input_tps_param, ThinPlateSpline
from software.model import Model
from software.dataset import get_dataset
from software.utils import LoggingParent
from software.ops import total_loss, PerceptualVGG
from software.metrics import ssim,psnr
from software.visualize import make_img_grid


WANDB_DISABLE_CODE = True

# from Tests import ThinPlateSpline1
# import tensorflow as tf


class PartBased(LoggingParent):

    def __init__(self, config, dirs):
        super().__init__()
        self.dirs = dirs
        self.config = config
        self.device = self.config.gpu

        self.is_debug = self.config.debug
        if self.is_debug:
            self.logger.info("Running in debug mode")

        if self.config.restart:
            self.logger.info(f'Resume training run with name "{self.config.name}" on device(s) {self.device}')
        else:
            self.logger.info(f'Start new training run with name "{self.config.name}" on device(s) {self.device}')

        ########## seed setting ##########
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        # random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(config["seed"])
        rng = np.random.RandomState(config["seed"])

        self.perc_loss = self.config.vgg_loss if hasattr(self.config,"vgg_loss") else False

        if self.config.mode == "train":
            wandb.init(
                dir=self.dirs["log"],
                project="vda_tp1-5",
                name=self.config.name,
            )

            self.logger.info("Training parameters:")
            for key in self.config.toDict():
                self.logger.info(f"{key}: {self.config[key]}")  # print to console
                wandb.config.update({key: self.config[key]})


    def _load_ckpt(self, key, dir=None,name=None, use_best=False):
        if dir is None:
            dir = self.dirs["ckpt"]

        if name is None:
            if len(os.listdir(dir)) > 0:
                ckpts = glob(path.join(dir,"*.pth"))

                # load latest stored checkpoint
                ckpts = [ckpt for ckpt in ckpts if key in ckpt.split("/")[-1]]
                if len(ckpts) == 0:
                    self.logger.info(f"*************No ckpt found****************")
                    op_ckpt = mod_ckpt = None
                    return mod_ckpt, op_ckpt
                if use_best:
                    ckpts = [x for x in glob(path.join(dir,"*.pt")) if "=" in x.split("/")[-1]]
                    ckpts = {float(x.split("=")[-1].split(".")[0]): x for x in ckpts}

                    ckpt = torch.load(
                        ckpts[max(list(ckpts.keys()))], map_location="cpu"
                    )
                else:
                    ckpts = {float(x.split("_")[-1].split(".")[0]): x for x in ckpts}

                    ckpt = torch.load(
                        ckpts[max(list(ckpts.keys()))], map_location="cpu"
                    )

                mod_ckpt = ckpt["model"] if "model" in ckpt else None
                key = [key for key in ckpt if key.startswith("optimizer")]
                assert len(key) == 1
                key = key[0]
                op_ckpt = ckpt[key]


                msg = "best model" if use_best else "model"

                if mod_ckpt is not None:
                    self.logger.info(f"*************Restored {msg} with key {key} from checkpoint****************")
                else:
                    self.logger.info(f"*************No ckpt for {msg} with key {key} found, not restoring...****************")

                if op_ckpt is not None:
                    self.logger.info(f"*************Restored optimizer with key {key} from checkpoint****************")
                else:
                    self.logger.info(f"*************No ckpt for optimizer with key {key} found, not restoring...****************")
            else:
                mod_ckpt = op_ckpt = None

            return mod_ckpt, op_ckpt


    def infer(self):
        pass
        # # Get args
        # bn = self.config.bn
        # mode = self.config.mode
        # name = self.config.name
        # load_from_ckpt = self.config.load_from_ckpt
        # lr = self.config.lr
        # epochs = self.config.epochs
        # device = torch.device('cuda:' + str(self.config.gpu) if torch.cuda.is_available() else 'cpu')
        # self.config.device = device
        # # Make Directory for Predictions
        # model_save_dir = '../results/' + name
        # if not os.path.exists(model_save_dir + '/predictions'):
        #     os.makedirs(model_save_dir + '/predictions')
        # # Load Model and Dataset
        # model = Model(self.config).to(device)
        # model = load_model(model, model_save_dir).to(device)
        # data = load_images_from_folder(stop=True)
        # test_data = np.array(data[-4:])
        # test_dataset = ImageDataset(test_data)
        # test_loader = DataLoader(test_dataset, batch_size=bn)
        # model.mode = 'predict'
        # model.eval()
        # # Predict on Dataset
        # for step, original in enumerate(test_loader):
        #     with torch.no_grad():
        #         original = original.to(device)
        #         tps_param_dic = tps_parameters(original.shape[0], self.config.scal, self.config.tps_scal, self.config.rot_scal, self.config.off_scal,
        #                                        self.config.scal_var, self.config.augm_scal)
        #         coord, vector = make_input_tps_param(tps_param_dic)
        #         coord, vector = coord.to(device), vector.to(device)
        #         # tf_coord, tf_vector = tf.convert_to_tensor(coord.cpu().detach().numpy()), tf.convert_to_tensor(vector.cpu().detach().numpy())
        #         # np_tensor = original.permute(0, 2, 3, 1).cpu().detach().numpy()
        #         # tf_tensor = tf.convert_to_tensor(np_tensor)
        #         # image_spatial_t, _ = ThinPlateSpline1(tf_tensor, tf_coord, tf_vector,
        #         #                                      256, 3)
        #         # image_spatial_t = torch.tensor(image_spatial_t.numpy()).permute(0, 3, 1, 2).to(device)
        #         image_spatial_t, _ = ThinPlateSpline(original, coord, vector,
        #                                              original.shape[3], device)
        #         image_appearance_t = K.ColorJitter(self.config.brightness, self.config.contrast, self.config.saturation, self.config.hue)(original)
        #         # image_spatial_t, image_appearance_t = transforms.ToTensor()(image_spatial_t.numpy()), \
        #         #                                       transforms.ToTensor()(image_appearance_t.numpy())
        #         image, reconstruction, mu, shape_stream_parts, heat_map = model(original, image_spatial_t,
        #                                                                         image_appearance_t, coord, vector)
        #         save_image(image[0], model_save_dir + '/predictions/original.png')
        #         save_image(reconstruction[0], model_save_dir + '/predictions/reconstruction.png')
        #         save_image(image_spatial_t[0], model_save_dir + '/predictions/spatial_transform.png')
        #         save_image(image_appearance_t[0], model_save_dir + '/predictions/appearance_transform.png')
        #         plot_tensor(original[0])
        #         plt.show()
        #         plot_tensor(image_spatial_t[0])
        #         plt.show()
        #         plot_tensor(image_appearance_t[0])
        #         plt.show()
        #         print(torch.max(image_spatial_t[0][0]), torch.min(image_spatial_t[0][0]))
        #         print(torch.max(image_spatial_t[0][1]), torch.min(image_spatial_t[0][1]))
        #         print(torch.max(image_spatial_t[0][2]), torch.min(image_spatial_t[0][2]))
        #         # print(mu[0][0], mu[0][1], mu[0][2], mu[0][3])
        #         save_heat_map(heat_map[0], model_save_dir)
        #         # for i in range(len(image)):
        #         #     save_image(image[i], model_save_dir + '/predictions/original' + str(i) + '.png')
        #         #     save_image(reconstruction[i],
        #         #                model_save_dir + '/predictions/reconstruction' + str(i) + '.png')

    def train(self):
        # Get self.configs
        bn = self.config.bn
        name = self.config.name
        load_from_ckpt = self.config.load_from_ckpt
        lr = self.config.lr
        epochs = self.config.epochs
        wd = self.config.weight_decay

        if self.config.restart and not self.is_debug:
            mod_ckpt, op_ckpt = self._load_ckpt("reg_ckpt")
        else:
            mod_ckpt = op_ckpt = None

        # get datasets for training and testing
        def w_init_fn(worker_id):
            return np.random.seed(np.random.get_state()[1][0] + worker_id)

        # Load Datasets and DataLoader
        dset = get_dataset(self.config.dataset)
        transforms = T.ToTensor()
        train_dataset = dset(self.config, transforms, train=True)
        test_dataset = dset(self.config, transforms, train=False)

        train_loader = DataLoader(train_dataset, batch_size=bn, shuffle=True,
                                  num_workers=0 if self.is_debug else self.config.n_workers, worker_init_fn=w_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=bn, shuffle=True,
                                 num_workers=0 if self.is_debug else self.config.n_workers, worker_init_fn=w_init_fn)
        eval_loader = DataLoader(test_dataset, batch_size=bn,
                                 num_workers=0 if self.is_debug else self.config.n_workers, worker_init_fn=w_init_fn)


        # model
        model = Model(self.config)
        self.logger.info(f"Number of trainable parameters in model is {sum(p.numel() for p in model.parameters())}")
        if self.config.restart and mod_ckpt is not None:
            self.logger.info("Load pretrained parameters and resume training.")
            model.load_state_dict(mod_ckpt)

        model.cuda(self.device)

        wandb.watch(model, log="all")

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        if self.config.restart and op_ckpt is not None:
            self.logger.info("Load state_dict of optimizer.")
            optimizer.load_state_dict(op_ckpt)

        if self.perc_loss:
            self.vgg = PerceptualVGG()
            self.vgg.cuda(self.device)
        else:
            self.vgg = None

        n_epoch_train = self.config.epochs
        start_it = 0
        start_epoch = 0
        if self.config.restart and op_ckpt is not None:
            start_it = list(optimizer.state_dict()["state"].values())[-1]["step"]
            start_epoch = int(np.floor(start_it / len(train_loader)))
            assert self.config.epochs > start_epoch
            n_epoch_train = self.config.epochs - start_epoch


        def train_step(engine, batch):
            model.train()
            original = batch["images"].cuda(self.device)

            tps_param_dic = tps_parameters(original.shape[0], self.config.scal, self.config.tps_scal, self.config.rot_scal,
                                           self.config.off_scal, self.config.scal_var, self.config.augm_scal)
            coord, vector = make_input_tps_param(tps_param_dic)
            coord, vector = coord.cuda(self.device), vector.cuda(self.device)
            image_spatial_t, _ = ThinPlateSpline(original, coord, vector,
                                                 original.shape[3], self.device)
            image_appearance_t = K.ColorJitter(self.config.brightness, self.config.contrast, self.config.saturation, self.config.hue)(original)
            # Zero out gradients

            rec, ssp, asp, mu, heat_map = model(original, image_spatial_t, image_appearance_t, coord, vector)

            loss, rec_loss, equiv_loss = total_loss(original, rec, ssp, asp, mu, coord, vector,
                                                  self.device, self.config.L_mu, self.config.L_cov,
                                                  self.config.scal, self.config.l_2_scal, self.config.l_2_threshold, self.vgg)


            # fixme compute keypoint metrics if available

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_dict = {"loss": loss.item(), "rec_loss": rec_loss.item(), "equiv_loss": equiv_loss.item()}

            return out_dict

        def eval_step(engine,batch):
            model.eval()
            with torch.no_grad():
                original = batch["images"].cuda(self.device)

                tps_param_dic = tps_parameters(original.shape[0], self.config.scal, self.config.tps_scal, self.config.rot_scal,
                                               self.config.off_scal, self.config.scal_var, self.config.augm_scal)
                coord, vector = make_input_tps_param(tps_param_dic)
                coord, vector = coord.cuda(self.device), vector.cuda(self.device)
                image_spatial_t, _ = ThinPlateSpline(original, coord, vector,
                                                     original.shape[3], self.device)
                image_appearance_t = K.ColorJitter(self.config.brightness, self.config.contrast, self.config.saturation, self.config.hue)(original)
                # Zero out gradients

                rec, ssp, asp, mu, heat_map = model(original, image_spatial_t, image_appearance_t, coord, vector)

                loss, rec_loss, equiv_loss = total_loss(original, rec, ssp, asp, mu, coord, vector,
                                                        self.device, self.config.L_mu, self.config.L_cov,
                                                        self.config.scal, self.config.l_2_scal, self.config.l_2_threshold)

            metric_ssim = ssim(original,rec)
            metric_psnr = psnr(original,rec)
            # fixme keypoint metrics

            return {"loss": loss.item(), "rec_loss": rec_loss.item(), "equiv_loss": equiv_loss.item(), "ssim": float(metric_ssim), "psnr": float(metric_psnr)}




        def eval_visual(engine, eval_batch):
            model.eval()
            with torch.no_grad():
                original = eval_batch["images"].cuda(self.device)

                tps_param_dic = tps_parameters(original.shape[0], self.config.scal, self.config.tps_scal, self.config.rot_scal,
                                               self.config.off_scal, self.config.scal_var, self.config.augm_scal)
                coord, vector = make_input_tps_param(tps_param_dic)
                coord, vector = coord.cuda(self.device), vector.cuda(self.device)
                image_spatial_t, _ = ThinPlateSpline(original, coord, vector,
                                                     original.shape[3], self.device)
                image_appearance_t = K.ColorJitter(self.config.brightness, self.config.contrast, self.config.saturation, self.config.hue)(original)
                # Zero out gradients

                rec, ssp, asp, mu, heat_map = model(original, image_spatial_t, image_appearance_t, coord, vector)

            img_grid = make_img_grid(image_appearance_t, image_spatial_t, rec, original,mus=mu, n_logged=6)


            wandb.log({"Evaluation image logs": wandb.Image(img_grid, caption=f"Image logs on test set.")})

        self.logger.info("Initialize engines...")
        trainer = Engine(train_step)
        evaluator = Engine(eval_step)
        test_img_generator = Engine(eval_visual)
        self.logger.info("Finish engine initialization...")


        # checkpointing
        n_saved = 10
        ckpt_handler = ModelCheckpoint(self.dirs["ckpt"], "reg_ckpt", n_saved=n_saved, require_empty=False)
        save_dict = {"model": model, "optimizer": optimizer}
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=self.config.ckpt_intervall),
                                  ckpt_handler,
                                  save_dict)

        pbar = ProgressBar(ascii=True)
        pbar.attach(trainer, output_transform=lambda x: x)
        pbar.attach(evaluator, output_transform=lambda x: x)

        @trainer.on(Events.ITERATION_COMPLETED(every=self.config.log_intervall))
        def log(engine):
            it = engine.state.iteration
            wandb.log({"iteration": it})

            # log losses
            for key in engine.state.output:
                wandb.log({key: engine.state.output[key]})

            batch = engine.state.batch
            model.eval()

            original = batch["images"].cuda(self.device)

            with torch.no_grad():

                tps_param_dic = tps_parameters(original.shape[0], self.config.scal, self.config.tps_scal, self.config.rot_scal,
                                               self.config.off_scal, self.config.scal_var, self.config.augm_scal)
                coord, vector = make_input_tps_param(tps_param_dic)
                coord, vector = coord.cuda(self.device), vector.cuda(self.device)
                image_spatial_t, _ = ThinPlateSpline(original, coord, vector,
                                                     original.shape[3], self.device)
                image_appearance_t = K.ColorJitter(self.config.brightness, self.config.contrast, self.config.saturation, self.config.hue)(original)

                rec, ssp, asp, mu, heat_map = model(original, image_spatial_t, image_appearance_t, coord, vector)

            img_grid = make_img_grid(image_appearance_t,image_spatial_t,rec,original, mus=mu, n_logged=6)

            wandb.log({"Training image logs": wandb.Image(img_grid, caption=f"Image logs after {it} train steps.")})


        # metrics for training
        Average(output_transform=lambda x : x["loss"]).attach(trainer, "loss-epoch_avg")
        Average(output_transform=lambda x: x["rec_loss"]).attach(trainer, "rec_loss-epoch_avg")
        Average(output_transform=lambda x: x["equiv_loss"]).attach(trainer, "equiv_loss-epoch_avg")

        # metrics during evaluation
        Average(output_transform=lambda x : x["loss"]).attach(evaluator, "loss-eval")
        Average(output_transform=lambda x: x["rec_loss"]).attach(evaluator, "rec_loss-eval")
        Average(output_transform=lambda x: x["equiv_loss"]).attach(evaluator, "equiv_loss-eval")
        Average(output_transform=lambda x: x["psnr"]).attach(evaluator, "psnr-eval")
        Average(output_transform=lambda x: x["ssim"]).attach(evaluator, "ssim-eval")


        @trainer.on(Events.EPOCH_COMPLETED(every=self.config.metric_at_epochs))
        def metrics(engine):
            self.logger.info(f"Computing metrics after epoch #{engine.state.epoch}")
            batch_size = eval_loader.batch_size
            bs = 20 if self.is_debug else (int(8000 / batch_size) if len(test_dataset) > 8000 else len(eval_loader))
            evaluator.run(eval_loader, max_epochs=1, epoch_length=bs)
            [wandb.log({key: evaluator.state.metrics[key]}) for key in evaluator.state.metrics]

        @trainer.on(Events.ITERATION_COMPLETED(every=self.config.test_img_intervall))
        def make_test_grid(engine):
            test_img_generator.run(test_loader, max_epochs=1, epoch_length=1)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_train_avg(engine):
            wandb.log({"epoch": engine.state.epoch})
            [wandb.log({key: engine.state.metrics[key]}) for key in engine.state.metrics]

        @trainer.on(Events.STARTED)
        def set_start_it(engine):
            self.logger.info(f'Engine starting from iteration {start_it}, epoch {start_epoch}')
            engine.state.iteration = start_it
            engine.state.epoch = start_epoch

        # run everything
        n_step_per_epoch = 10 if self.is_debug else len(train_loader)
        self.logger.info("Start training...")
        trainer.run(train_loader, max_epochs=n_epoch_train, epoch_length=n_step_per_epoch)
        self.logger.info("End training.")




        # # Make Training
        # for epoch in range(epochs+1):
        #     # # Train on Train Set
        #     # model.train()
        #     # model.mode = 'train'
        #     # for step, original in enumerate(train_loader):
        #     #     original = original.to(device)
        #     #     # Make transformations
        #     #     tps_param_dic = tps_parameters(original.shape[0], self.config.scal, self.config.tps_scal, self.config.rot_scal,
        #     #                                    self.config.off_scal, self.config.scal_var, self.config.augm_scal)
        #     #     coord, vector = make_input_tps_param(tps_param_dic)
        #     #     coord, vector = coord.to(device), vector.to(device)
        #     #     image_spatial_t, _ = ThinPlateSpline(original, coord, vector,
        #     #                                          original.shape[3], device)
        #     #     image_appearance_t = K.ColorJitter(self.config.brightness, self.config.contrast, self.config.saturation, self.config.hue)(original)
        #     #     # Zero out gradients
        #     #
        #     #     prediction, loss = model(original, image_spatial_t, image_appearance_t, coord, vector)
        #     #
        #     #     optimizer.zero_grad()
        #     #     loss.backward()
        #     #     optimizer.step()
        #     #     if step == 0:
        #     #         loss_log = torch.tensor([loss])
        #     #     else:
        #     #         loss_log = torch.cat([loss_log, torch.tensor([loss])])
        #     # print(f'Epoch: {epoch}, Train Loss: {torch.mean(loss_log)}')
        #
        #     # Evaluate on Test Set
        #     model.eval()
        #     for step, original in enumerate(test_loader):
        #         with torch.no_grad():
        #             original = original.to(device)
        #             tps_param_dic = tps_parameters(original.shape[0], self.config.scal, self.config.tps_scal, self.config.rot_scal, self.config.off_scal,
        #                                            self.config.scal_var, self.config.augm_scal)
        #             coord, vector = make_input_tps_param(tps_param_dic)
        #             coord, vector = coord.to(device), vector.to(device)
        #             image_spatial_t, _ = ThinPlateSpline(original, coord, vector,
        #                                                  original.shape[3], device)
        #             image_appearance_t = K.ColorJitter(self.config.brightness, self.config.contrast, self.config.saturation, self.config.hue)(original)
        #             prediction, loss = model(original, image_spatial_t, image_appearance_t, coord, vector)
        #             if step == 0:
        #                 loss_log = torch.tensor([loss])
        #             else:
        #                 loss_log = torch.cat([loss_log, torch.tensor([loss])])
        #     print(f'Epoch: {epoch}, Test Loss: {torch.mean(loss)}')
        #
        #     # Track Progress
        #     if epoch % 5 == 0:
        #         model.mode = 'predict'
        #         image, reconstruction, mu, shape_stream_parts, heat_map = model(original, image_spatial_t,
        #                                                               image_appearance_t, coord, vector)
        #         for i in range(len(image)):
        #             if epoch == 0:
        #                 save_image(image[i], model_save_dir + '/image/' + str(i) + '_' + str(epoch) + '.png')
        #             save_image(reconstruction[i], model_save_dir + '/reconstruction/' + str(i) + '_' + str(epoch) + '.png')
        #             # save_image(shape_stream_parts[0][0],
        #             #            model_save_dir + '/parts/' + str(i) + '_' + str(epoch) + '.png')
        #             # save_image(heat_map[0][0],
        #             #            model_save_dir + '/heat_map/' + str(i) + '_' + str(epoch) + '.png')
        #         save_model(model, model_save_dir)



# if __name__ == '__main__':
#     arg = DotMap(vars(parse_args()))
#     main(arg)



# def main2(arg):
#     # Get args
#     bn = arg.bn
#     mode = arg.mode
#     name = arg.name
#     load_from_ckpt = arg.load_from_ckpt
#     lr = arg.lr
#     weight_decay = arg.weight_decay
#     epochs = arg.epochs
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     arg.device = device
#
#     if mode == 'train':
#         # Make new directory
#         model_save_dir = name + "/training2"
#         if not os.path.exists(model_save_dir):
#             os.makedirs(model_save_dir)
#             os.makedirs(model_save_dir + '/image')
#             os.makedirs(model_save_dir + '/reconstruction')
#             os.makedirs(model_save_dir + '/mu')
#             os.makedirs(model_save_dir + '/parts')
#
#         # Load Datasets
#         train_data = load_images_from_folder()[:100]
#         train_dataset = ImageDataset2(train_data, arg)
#         test_data = load_images_from_folder()[-1000:]
#         test_dataset = ImageDataset2(test_data, arg)
#
#         # Prepare Dataloader & Instances
#         train_loader = DataLoader(train_dataset, batch_size=bn, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=bn)
#         model = Model2(arg).to(device)
#         if load_from_ckpt == True:
#             model = load_model(model, model_save_dir).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#         # Make Training
#         for epoch in range(epochs):
#             # Train on Train Set
#             model.train()
#             model.mode = 'train'
#             for step, original in enumerate(train_loader):
#                 original = original.to(device, dtype=torch.float)
#                 optimizer.zero_grad()
#                 # plot_tensor(original[0])
#                 # plot_tensor(spat[0])
#                 # plot_tensor(app[0])
#                 # plot_tensor(original[1])
#                 # plot_tensor(spat[1])
#                 # plot_tensor(app[1])
#                 # print(coord, vec)
#                 prediction, loss = model(original)
#                 loss.backward()
#                 optimizer.step()
#                 if epoch % 2 == 0 and step == 0:
#                     print(f'Epoch: {epoch}, Train Loss: {loss}')
#
#             # Evaluate on Test Set
#             model.eval()
#             for step, original in enumerate(test_loader):
#                 with torch.no_grad():
#                     original = original.to(device, dtype=torch.float)
#                     prediction, loss = model(original)
#                     if epoch % 2 == 0 and step == 0:
#                         print(f'Epoch: {epoch}, Test Loss: {loss}')
#
#             # Track Progress
#             if epoch % 5 == 0:
#                 model.mode = 'predict'
#                 image, reconstruction, mu, shape_stream_parts, heat_map = model(original)
#                 for i in range(len(image)):
#                     save_image(image[i], model_save_dir + '/image/' + str(i) + '_' + str(epoch) + '.png')
#                     save_image(reconstruction[i], model_save_dir + '/image/' + str(i) + '_' + str(epoch) + '.png')
#                     #save_image(mu[i], model_save_dir + '/image/' + str(epoch) + '.png')
#                     #save_image(shape_stream_parts[i], model_save_dir + '/image/' + str(epoch) + '.png')
#
#             # Save the current Model
#             if epoch % 50 == 0:
#                 save_model(model, model_save_dir)
#
#     elif arg.mode == 'predict':
#         model_save_dir = arg.name + "/prediction"
#         if not os.path.exists(model_save_dir):
#             os.makedirs(model_save_dir)

