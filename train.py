import argparse
import os
from datetime import datetime

import gin
import pytorch_lightning as pl

from src.models import get_model
from src.data import get_data_module
from src.modules import get_lightning_module
from src.utils.file import ensure_dir
from src.utils.logger import setup_logger
from src.utils.misc import logged_hparams


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
    gin.finalize()

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
        additional_kwargs["sync_batchnorm"] = False
        # additional_kwargs["strategy"] = "ddp_find_unused_parameters_false"
        additional_kwargs["strategy"] = "ddp"
        additional_kwargs["accelerator"]="gpu"
        additional_kwargs["precision"]=16

    trainer = pl.Trainer(
        default_root_dir=save_path,
        max_epochs=max_epoch,
        max_steps=max_step,
        gpus=gpus,
        callbacks=callbacks,
        # logger=loggers,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        **additional_kwargs
    )

    # write config file
    with open(os.path.join(save_path, "config.gin"), "w") as f:
        f.write(gin.operative_config_str())

    trainer.fit(pl_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--save_path", type=str, default="experiments")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1235)
    parser.add_argument("-v", "--voxel_size", type=float, default=None)
    parser.add_argument("-v", "--gpus", type=int, default=1)
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
        gin.query_parameter("train.model_name") + '_' + gin.query_parameter("DimensionlessCoordinates.voxel_size")[2:]
    )
    # setup_logger(args.run_name, args.debug)

    train(save_path=save_path, run_name=args.run_name)