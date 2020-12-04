import argparse
from os import path, makedirs
import torch
import yaml
import os
from dotmap import DotMap

from software.experiment import PartBased

def create_dir_structure(config):
    subdirs = ["ckpt", "config", "generated", "log"]
    structure = {subdir: path.join(config["base_dir"],subdir,config["name"]) for subdir in subdirs}
    if "DATAPATH" in os.environ:
        structure = {subdir: path.join(os.environ["DATAPATH"],structure[subdir]) for subdir in structure}
    return structure

def load_parameters(config_name, restart,debug,input_dir=None, output_dir=None):
    with open(config_name,"r") as f:
        cdict = yaml.load(f,Loader=yaml.FullLoader)
    # if we just want to test if it runs
    cdict["debug"] = debug
    if debug:
        cdict["name"] = "debug"
    if output_dir is not None:
        cdict["base_dir"] = output_dir
    if input_dir is not None:
        cdict["datapath"] = input_dir

    dir_structure = create_dir_structure(cdict)
    saved_config = path.join(dir_structure["config"], "config.yaml")
    if restart:
        if path.isfile(saved_config):
            with open(saved_config,"r") as f:
                cdict = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise FileNotFoundError("No saved config file found but model is intended to be restarted. Aborting....")

    else:
        [makedirs(dir_structure[d],exist_ok=True) for d in dir_structure]
        if path.isfile(saved_config) and not cdict["debug"]:
            print(f"\033[93m" + "WARNING: Model has been started somewhen earlier: Resume training (y/n)?" + "\033[0m")
            while True:
                answer = input()
                 # answer = "n"
                if answer == "y" or answer == "yes":
                    with open(saved_config,"r") as f:
                        cdict = yaml.load(f, Loader=yaml.FullLoader)

                    restart = True
                    break
                elif answer == "n" or answer == "no":
                    with open(saved_config, "w") as f:
                        yaml.dump(cdict, f, default_flow_style=False)
                    break
                else:
                    print(f"\033[93m" + "Invalid answer! Try again!(y/n)" + "\033[0m")
        else:
            with open(saved_config, "w") as f:
                yaml.dump(cdict,f,default_flow_style=False)

    return cdict, dir_structure, restart


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        default="config.yaml",
                        help="Define config file")
    parser.add_argument("-r","--restart", default=False,action="store_true",help="Whether training should be resumed.")
    parser.add_argument("--gpu",default=0, type=int,help="GPU to use.")
    parser.add_argument("--debug","-d",default=False, action="store_true", help="Whether or not to run the script in debug mode.")
    parser.add_argument("-m","--mode",default="train",type=str,choices=["train","inference"],help="Whether to start in train or infer mode?")
    parser.add_argument("--input",type=str, default=None, help="The input-, or data-path.")
    parser.add_argument("--output", type=str, default=None, help="The path were the output should be stored, including logs etc.")


    args = parser.parse_args()

    args.debug = args.debug and (args.mode == "train")


    config, structure, restart = load_parameters(args.config, args.restart or args.mode == "test",args.debug,args.input, args.output)
    config["restart"] = restart
    config["mode"] = args.mode

    gpu = torch.device(
        f"cuda:{int(args.gpu)}"
        if torch.cuda.is_available() and int(args.gpu) >= 0
        else "cpu"
    )

    config["gpu"]= gpu
    config = DotMap(config)
    partbased_experiment = PartBased(config, structure)
    if config["mode"] == "train":
        partbased_experiment.train()
    elif config["mode"] == "inference":
        partbased_experiment.infer()
    else:
        raise ValueError(f'"mode"-parameter should be either "train" or "infer" but is actually {config["mode"]}')