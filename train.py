import argparse
import os
from datetime import datetime

import gin
import pytorch_lightning as pl
# import lightning.pytorch as pl

from src.models import get_model
from src.data import get_data_module
from src.modules import get_lightning_module
from src.utils.file import ensure_dir
from src.utils.logger import setup_logger
from src.utils.misc import logged_hparams

from src.models.resunet import Res16UNetBase

import MinkowskiEngine as ME
import torch

# from lightning.pytorch.strategies import FSDPStrategy
from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp import FullStateDictConfig

def _strip_prefix_from_state_dict(k, prefix):
    return k[len(prefix) :] if k.startswith(prefix) else k


# class CustomFSDP(FSDPStrategy):
#     def lightning_module_state_dict(self):
#         """Gathers the full state dict by unsharding all the parameters.
#         To avoid OOM, the returned parameters will only be returned on rank 0 and on CPU. All other ranks get an empty
#         dict.
#         """
#         assert self.model is not None

#         with FullyShardedDataParallel.state_dict_type(
#             module=self.model,
#             state_dict_type=StateDictType.FULL_STATE_DICT,
#             state_dict_config=FullStateDictConfig(offload_to_cpu=False,
#                                                   # offload_to_cpu=(self.world_size > 1),
#                                                   rank0_only=True),):
#         # with FullyShardedDataParallel.state_dict_type(
#         #     module=self.model,
#         #     state_dict_type=StateDictType.FULL_STATE_DICT,
#         #     state_dict_config=FullStateDictConfig(offload_to_cpu=(self.world_size > 1),
#         #                                           rank0_only=False),):
#             # state_dict = self.model.state_dict()
#             # print('=======================')
#             # print(state_dict.keys())
#             # return {_strip_prefix_from_state_dict(k, "_forward_module."): v for k, v in ckpt["state_dict"].items()}
#             return self.model.state_dict()
        
# # # fsdp = CustomFSDP(
fsdp = DDPFullyShardedNativeStrategy(
            activation_checkpointing=[
                ME.MinkowskiConvolution,
                ME.MinkowskiConvolutionTranspose,
                ME.MinkowskiReLU,
                ME.MinkowskiBatchNorm,
                ME.MinkowskiDropout,
                ME.MinkowskiSumPooling
            ],
            # or pass a list with multiple types
        )


@gin.configurable
def train(
    save_path,
    project_name,
    run_name,
    lightning_module_name,
    data_module_name,
    model_name,
    gpus,
    log_every_n_steps,
    check_val_every_n_epoch,
    refresh_rate_per_second,
    best_metric,
    max_epoch,
    max_step,
    augs,
):
    now = datetime.now().strftime('%m-%d-%H-%M-%S')
    run_name = run_name + "_" + now
    # save_path = os.path.join(save_path, run_name)
    ensure_dir(save_path)

    ## get data and model
    data_module = get_data_module(data_module_name)()
    model = get_model(model_name)()
    if gpus > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    pl_module = get_lightning_module(lightning_module_name)(model=model, max_steps=max_step)
    # gin.finalize()

    hparams = logged_hparams()
    callbacks = [
        pl.callbacks.TQDMProgressBar(refresh_rate=refresh_rate_per_second),
        pl.callbacks.ModelCheckpoint(
            dirpath=save_path, monitor=best_metric, save_last=True, save_top_k=1, mode="max"
        ),
        pl.callbacks.LearningRateMonitor(),
    ]
    # loggers = [
    #     pl.loggers.WandbLogger(
    #         name=run_name,
    #         save_dir=save_path,
    #         project=project_name,
    #         log_model=True,
    #         entity="chrockey",
    #         config=hparams,
    #     )
    # ]
    additional_kwargs = dict()
    if gpus > 1:
        additional_kwargs["replace_sampler_ddp"] = True
        additional_kwargs["sync_batchnorm"] = False ## little influence to memory usage
        # additional_kwargs["find_unused_parameters"]=False
        additional_kwargs["strategy"] = "ddp_find_unused_parameters_false"
        # additional_kwargs["strategy"] = "ddp_sharded"
        # additional_kwargs["strategy"] = fsdp
        additional_kwargs["accelerator"]="gpu"
        additional_kwargs["precision"]=32
        # additional_kwargs["num_nodes"]=gpus

    # write config file
    with open(os.path.join(save_path, "config.gin"), "w") as f:
        f.write(gin.operative_config_str())

    if augs:
        resolutions = [0.02, 0.05, 0.10, 0.15]
        per_step = 4800
        ckpt_path = None

        trainer = pl.Trainer(
                    default_root_dir=save_path,
                    max_epochs=max_epoch,
                    # max_steps=max_step,
                    # max_steps=per_step*(i+j+1),
                    max_steps=0,
                    gpus=gpus,
                    devices=gpus,
                    callbacks=callbacks,
                    # logger=loggers,
                    log_every_n_steps=log_every_n_steps,
                    check_val_every_n_epoch=check_val_every_n_epoch,
                    num_sanity_val_steps=0,
                    # ckpt_path=ckpt_path
                    **additional_kwargs
                )
        
        for i in range(int(max_step/(per_step*4))):
            for j in range(len(resolutions)):
                gin.bind_parameter("DimensionlessCoordinates.voxel_size", resolutions[j])
                print(gin.query_parameter("DimensionlessCoordinates.voxel_size"))

                trainer.fit_loop.epoch_loop.max_steps += per_step
                trainer.fit(pl_module, data_module)
                # ckpt_path = os.path.join(save_path, 'last.ckpt')
    else:
        # print('---------------')
        trainer = pl.Trainer(
                    default_root_dir=save_path,
                    max_epochs=max_epoch,
                    max_steps=max_step,
                    gpus=gpus,
                    devices=gpus,
                    callbacks=callbacks,
                    # logger=loggers,
                    log_every_n_steps=log_every_n_steps,
                    check_val_every_n_epoch=check_val_every_n_epoch,
                    num_sanity_val_steps=0,
                    **additional_kwargs
                )
        trainer.fit(pl_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--save_path", type=str, default="experiments")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1235)
    parser.add_argument("-v", "--voxel_size", type=float, default=None)
    parser.add_argument("-g", "--gpus", type=int, default=1)
    parser.add_argument("-a", "--augumentation", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    gin.parse_config_file(args.config)
    if args.voxel_size is not None:
        gin.bind_parameter("DimensionlessCoordinates.voxel_size", args.voxel_size)
    if args.gpus > 1:
        gin.bind_parameter("train.gpus", args.gpus)
        gin.bind_parameter("LitSegmentationModuleBase.lr", gin.query_parameter("LitSegmentationModuleBase.lr")*args.gpus)
    
    save_path = os.path.join(
        args.save_path,
        gin.query_parameter("train.model_name") + '_' + str(gin.query_parameter("DimensionlessCoordinates.voxel_size"))[2:]
    )
    # setup_logger(args.run_name, args.debug)

    train(save_path=save_path, run_name=args.run_name, augs = args.augumentation)