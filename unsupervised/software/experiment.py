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
from kornia.enhance import normalize_min_max
import wandb

from software.transformations import tps_parameters, make_input_tps_param, ThinPlateSpline
from software.model import LandmarkModel
from software.dataset import get_dataset
from software.utils import LoggingParent
from software.ops import PerceptualVGG
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

    def train(self):
        # Get self.configs
        bn = self.config.bn
        lr = self.config.lr
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
        model = LandmarkModel(self.config)
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
            img_rec, rec_same_id, loss, rec_loss, transform_loss, precision_loss, mu, L_inv, part_maps_raw = model(original)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            equiv_loss = self.config.L_mu * transform_loss + self.config.L_cov * precision_loss
            report_means = torch.mean(torch.norm(mu, p=1, dim=2), dim=0).cpu().detach().numpy()
            report_prec = torch.mean(torch.linalg.norm(L_inv, ord='fro', dim=[2, 3]),dim=0).cpu().detach().numpy()

            mean_dict = {f"part_{pc}-mu-norm": l.item() for pc,l in enumerate(report_means)}
            prec_dict = {f"part_{pc}-prec-norm": c.item() for pc,c in enumerate(report_prec)}

            out_dict = {"loss": loss.item(), "rec_loss": rec_loss.item(), "equiv_loss": equiv_loss.item(),
                        "transform_loss": transform_loss.item(), "precision_loss": precision_loss.item()}

            out_dict.update(mean_dict)
            out_dict.update(prec_dict)

            return out_dict

        def eval_step(engine,batch):
            model.eval()
            with torch.no_grad():
                original = batch["images"].cuda(self.device)
                img_rec, rec_same_id, loss, rec_loss, transform_loss, precision_loss, mu, L_inv, part_maps_raw = model(original)

                equiv_loss = self.config.L_mu * transform_loss + self.config.L_cov * precision_loss
                # report_means = torch.mean(torch.norm(mu, p=1, dim=2), dim=0).cpu().detach().numpy()
                # report_prec = torch.mean(torch.linalg.norm(L_inv, ord='fro', dim=[2, 3]), dim=0).cpu().detach().numpy()

            # mean_dict = {f"part_{pc}-mu-norm": l.item() for pc, l in enumerate(report_means)}
            # prec_dict = {f"part_{pc}-prec-norm": c.item() for pc, c in enumerate(report_prec)}

            out_dict = {"loss": loss.item(), "rec_loss": rec_loss.item(), "equiv_loss": equiv_loss.item(),
                        "transform_loss": transform_loss.item(), "precision_loss": precision_loss.item()}

            # out_dict.update(mean_dict)
            # out_dict.update(prec_dict)

            metric_ssim = ssim(original,rec_same_id[:original.shape[0]])
            metric_psnr = psnr(original,rec_same_id[:original.shape[0]])
            # fixme keypoint metrics
            out_dict.update({"ssim": float(metric_ssim), "psnr": float(metric_psnr)})

            return out_dict




        def eval_visual(engine, eval_batch):
            model.eval()
            with torch.no_grad():
                original = eval_batch["images"].cuda(self.device)

                img_rec, rec_same_id, loss, rec_loss, transform_loss, precision_loss, mu, L_inv, part_maps_raw = model(original)

                # get part maps for original image
                part_maps, mu = model.detect_parts(original)


            image_appearance_t = img_rec[original.shape[0]:]
            image_spatial_t = img_rec[:original.shape[0]]
            rec_same_id = rec_same_id[:original.shape[0]]
            img_grid = make_img_grid(image_appearance_t, image_spatial_t, rec_same_id, original, mus=mu, n_logged=6)


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
        pbar.attach(trainer, output_transform=lambda x: {key: x[key] for key in x if "part" not in key})
        pbar.attach(evaluator, output_transform=lambda x: x)

        @trainer.on(Events.ITERATION_COMPLETED(every=self.config.log_intervall))
        def log(engine):
            it = engine.state.iteration
            wandb.log({"iteration": it})

            # log losses
            for key in engine.state.output:
                wandb.log({key: engine.state.output[key]})

            batch = engine.state.batch

            original = batch["images"].cuda(self.device)

            with torch.no_grad():
                img_rec, rec_same_id, loss, rec_loss, transform_loss, precision_loss, mu, L_inv, part_maps_raw = model(original)
                part_maps, mu = model.detect_parts(original)


                image_appearance_t = img_rec[original.shape[0]:]
                image_spatial_t = img_rec[:original.shape[0]]
                rec_same_id = rec_same_id[:original.shape[0]]

            img_grid = make_img_grid(image_appearance_t,image_spatial_t,rec_same_id,original, mus=mu, n_logged=6)

            wandb.log({"Training image logs": wandb.Image(img_grid, caption=f"Image logs after {it} train steps.")})


        # metrics for training
        Average(output_transform=lambda x : x["loss"]).attach(trainer, "loss-epoch_avg")
        Average(output_transform=lambda x: x["rec_loss"]).attach(trainer, "rec_loss-epoch_avg")
        Average(output_transform=lambda x: x["equiv_loss"]).attach(trainer, "equiv_loss-epoch_avg")
        Average(output_transform=lambda x: x["transform_loss"]).attach(trainer, "transform_loss-epoch_avg")
        Average(output_transform=lambda x: x["precision_loss"]).attach(trainer, "precision_loss-epoch_avg")

        # metrics during evaluation
        Average(output_transform=lambda x : x["loss"]).attach(evaluator, "loss-eval")
        Average(output_transform=lambda x: x["rec_loss"]).attach(evaluator, "rec_loss-eval")
        Average(output_transform=lambda x: x["equiv_loss"]).attach(evaluator, "equiv_loss-eval")
        Average(output_transform=lambda x: x["transform_loss"]).attach(evaluator, "transform_loss-eval")
        Average(output_transform=lambda x: x["precision_loss"]).attach(evaluator, "precision_loss-eval")
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
