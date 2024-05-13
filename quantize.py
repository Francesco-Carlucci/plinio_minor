import torch

#pytorch 2 export quantization method
from torch.ao.quantization.quantize_pt2e import (
  prepare_pt2e,
  convert_pt2e,
)
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
  XNNPACKQuantizer,
  get_symmetric_quantization_config,
)

#PLiNIO MPS
from plinio.methods.mps import MPS, MPSType, get_default_qinfo
from plinio.cost import params_bit

import argparse
import omegaconf
import os
from core.utils import (print_criterion, get_bp_pk_vly_mask,
                        glob_dez, glob_z, glob_demm, glob_mm,
                        loc_dez, loc_z, loc_demm, loc_mm)
#dataset loading
from core.utils import (get_nested_fold_idx, mat2df)
from scipy.io import loadmat
#import sys
import numpy as np
import pandas as pd

#from core.solver_s2l import SolverS2l
#from core.solver_s2s import Solver
#from core.solver_f2l import SolverF2l as solver_f2l

from core.models import *
from core.models.unet1d import Unet1d
from core.models.resnet1d import Resnet1d

from core.models.trainer import MyTrainer
from core.loaders import *

from torchinfo import summary

def get_parser():
    parser = argparse.ArgumentParser()
    # general config
    parser.add_argument("--config_file", type=str, help="Path for the config file")

    return parser

def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            x, y, x_abp, peakmask, vlymask = batch
            self.model(x['ppg'])

def _get_loader(config):
        if config.exp.loader=="waveform":
            return WavDataModule(config)
        elif config.exp.loader=="mabp":
            return MABPDataModule(config)
        elif config.exp.loader=="feature":
            return FeatDataModule(config)


def main():

    """
    "input_size": 1,
    "output_size": 625,
    "output_channel": 8,
    "layers": [2,2,2,2],
    "sample_step": 1,
    """


    config={
        "exp":{
                "fs": 125,
                "N_fold": 5,
                "random_state": 100,
                "data_name": "bcg",
                "file_format": "mat",
                "model_type": "resnet1d",
                #"exp_name": ${exp.data_name}-${exp.model_type},
                #"data_dir": "../../benfenati/data_folder/PPG/BP", #datasets/augmented
                "subject_dict": "../../benfenati/data_folder/PPG/BP/bcg_dataset/signal_fold",
                "loader": "waveform",
                "cv": "cv"},
        "param_model":
                    {
                    "N_epoch": 256,
                    "batch_size": 256,
                    "lr": 0.001,

                    "in_channel": 1,
                    "base_filters": 32,
                    "first_kernel_size": 9,
                    "kernel_size": 5,
                    "stride": 4,
                    "groups": 1,
                    "n_block": 4,
                    "output_size": 2,
                    "sample_step": 1,
                    "is_se": False,
                    "se_ch_low": 4,

                    "use_plinio": "pit",
                    "plinio_strength": 1e-7,
                    "plinio_cost": "ops",
                    "lr_plinio": 0.05
                    },
        "param_trainer":{
                        "max_epochs": 10,
                        "check_val_every_n_epoch": 20,
                        "devices": [0],
                        "min_epochs": 50},
        "param_loader":{
                        "ppg_norm": "loc_z",
                        "bp_norm": "glob_mm"}
        }

    config=omegaconf.dictconfig.DictConfig(config)

    model_path=        "/root/Tesi/PPG-BP/code/train/unet_uci_supernet/unet1d_uci2_1/fold0/lightning_logs/version_2/checkpoints/end_tuning.ckpt" #epoch=19-val_mse=0.009
    before_export_path="/root/Tesi/PPG-BP/code/train/unet_uci_supernet/unet1d_uci2_1/fold0/lightning_logs/version_1/checkpoints/end_training.ckpt"
    #config = OmegaConf.load(args.config_file)

    input_shape=(config.param_model.batch_size,1,625) #self.config.param_model.batch_size,
    if "ppgbp" in config.exp.data_name:
        input_shape=(config.param_model.batch_size,1,262)

    if config.param_loader.bp_norm=='loc_z':
        bp_norm = loc_z
        bp_denorm = loc_dez
    elif config.param_loader.bp_norm=='loc_mm':
        bp_norm = loc_mm
        bp_denorm = loc_demm
    elif config.param_loader.bp_norm=='glob_z':
        bp_norm = glob_z
        bp_denorm = glob_dez
    elif config.param_loader.bp_norm=='glob_mm':
        bp_norm = glob_mm
        bp_denorm = glob_demm

    #for foldIdx in range(n_fold):

    loaded_model=torch.load(model_path, map_location="cuda")
    config.param_loader=loaded_model['hyper_parameters']['param_loader']
    config.param_model=loaded_model['hyper_parameters']['param_model']

    dm = _get_loader(config)

    #load dataset
    folds_train, folds_val, folds_test =list(get_nested_fold_idx(config.exp.N_fold))[0]
    #all_split_df = [mat2df(loadmat(f"{config.exp.subject_dict}_{i}.mat")) for i in [folds_train[0],folds_val[0]]]
    all_split_df = [mat2df(loadmat(f"{config.exp.subject_dict}_{i}.mat")) for i in range(config.exp.N_fold)]

    train_df = pd.concat(np.array(all_split_df, dtype=object)[folds_train]) #[0][0:10])
    val_df = pd.concat(np.array(all_split_df, dtype=object)[folds_val]) #[0][0:10])
    test_df = pd.concat(np.array(all_split_df, dtype=object)[folds_test]) #[0][0:10])

    dm.setup_kfold(train_df, val_df, test_df)

    #solver=SolverS2l(config)
    #if config.exp.model_type in ['unet1d', 'ppgiabp', 'vnet']:
    #    solver=Solver(config)
    #elif config.exp.model_type in ['resnet1d','spectroresnet','mlpbp','temponet']:
    #    solver=SolverS2l(config)
    #else:
    #    solver = solver_f2l(config)

    # model=solver._get_model(norm_func=bp_denorm,
    #                     param_loader=config.param_loader,logdir=None)

    model = Resnet1d(config.param_model,
           norm_func=dm.test_dataloader().dataset.bp_denorm,
           param_loader=config.param_loader,
           logdir=None,
           input_shape=input_shape)

    #model = Unet1d(config.param_model,
    #                norm_func=dm.test_dataloader().dataset.bp_denorm,
    #                param_loader=config.param_loader,
    #                logdir=None,
    #                input_shape=input_shape)

    model.plinize()
    model_before_export=torch.load(before_export_path, map_location="cuda")
    model.load_state_dict(model_before_export['state_dict'])#,strict=false)
    model.export()
    model.load_state_dict(loaded_model['state_dict'])
    model=model.eval()

    """ pytorch 2 export
    #export the model
    model_to_quantize=model.eval()
    example_inputs=torch.randn(input_shape)
    exported_model = capture_pre_autograd_graph(model_to_quantize, [example_inputs.cuda()])
    #prepare quantizer
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config())
    #prepare the model
    prepared_model = prepare_pt2e(exported_model, quantizer)
    #print(prepared_model.graph)
    """
    #plinize
    model.use_plinio_cost=True
    model.model = MPS(model.model,
            input_shape=input_shape[1:], #unbatched input shape
            qinfo=get_default_qinfo(
                w_precision=(8,),
                a_precision=(8,)),
            w_search_type=MPSType.PER_LAYER,
            cost=params_bit)

    trainer = MyTrainer(max_epochs=1, devices=[0], max_steps=10)

    trainer.fit(model, dm)

    #model.export()

    print(model)

    #verificare get_cost!=0
    #training come finetuning con rete MPS
    #



    #calibrate(prepared_model, val_df)



if __name__ == '__main__':
    #parser = get_parser()
    main() #parser.parse_args())