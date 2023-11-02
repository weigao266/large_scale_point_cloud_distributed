import argparse
import os
from datetime import datetime
import time

import gin
import pytorch_lightning as pl
from src.models import get_model
from src.data import get_data_module
from src.modules import get_lightning_module
from src.utils.file import ensure_dir
from src.utils.logger import setup_logger
from src.utils.misc import logged_hparams

from src.models.resunet import Res16UNetBase
from src.segmentation_trainer import SegmentationTrainer

import MinkowskiEngine as ME
import torch

from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp import FullStateDictConfig

def _strip_prefix_from_state_dict(k, prefix):
    return k[len(prefix) :] if k.startswith(prefix) else k

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
    resume_path,
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
    hparams = logged_hparams()

    if torch.cuda.is_available():
        # import pdb; pdb.set_trace()
        device = torch.device("cuda")
        model = model.to(device)

    # write config file
    with open(os.path.join(save_path, "config.gin"), "w") as f:
        f.write(gin.operative_config_str())

    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    trainer = SegmentationTrainer(model=model, max_steps=max_step, max_epoch=max_epoch, device=device)

    optimizer_lrscheduler = trainer.configure_optimizer_lrscheduler()

    init_epoch = 0
    if resume_path is not None:
        ckpt = torch.load(resume_path)
        trainer.model.load_state_dict(ckpt['model_state_dict'])
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        init_epoch = ckpt['epoch']

    # print(len(train_dataloader))
    batch_size = gin.query_parameter("ScanNetRGBDataModule.train_batch_size")
    max_epoch = max_epoch if max_epoch != 0 else int(max_step/(len(train_dataloader)/batch_size))
    trainer.max_epoch = max_epoch
    for epoch in range(init_epoch, max_epoch):
        # train for one epoch
        # import pdb; pdb.set_trace()
        train_one_epoch(hparams, trainer, train_dataloader, epoch + init_epoch, max_epoch, save_path, device)

        if check_val_every_n_epoch != 0 and (epoch % check_val_every_n_epoch == 0):
            # print(len(val_dataloader))
            val_one_epoch(hparams, trainer, val_dataloader, epoch + init_epoch, save_path, device)


def val_one_epoch(hparams, trainer, dataloader, epoch, save_path, device):
    mloss = 0
    for i, batch in enumerate(dataloader):

        # move data to the same device as model
        # import pdb; pdb.set_trace()
        loss = trainer.val_one_step(batch)

        # print("Iterations:", str(i),
        #       "Train Loss", loss.item(),
        #       )

        mloss += loss.item()
        del loss

    print("Epoch val loss:", str(mloss/(i+1)))
    
    trainer.on_validation_epoch_end(epoch, save_path)

def train_one_epoch(hparams, trainer, train_dataloader, epoch, max_epoch, save_path, device):
    mloss = 0
    et = time.time()
    
    for i, batch in enumerate(train_dataloader):

        # move data to the same device as model
        # import pdb; pdb.set_trace()
        t0 = time.time()
        loss = trainer.train_one_step(batch)

        if epoch < 3:
            print("Iterations / Epochs / Total Epochs:", str(i), "/", str(epoch), "/", str(max_epoch),
                "Train Loss:", loss.item(), 
                "Batch Size:", batch["batch_size"], 
                "Batch Time:", str(time.time()-t0), 'seconds',
                )

        mloss += loss.item()
        del loss

    print("Epochs / Total Epoch:", str(epoch), "/", str(max_epoch),
          "Epoch train loss:", str(mloss/(i+1)), 
          "Epoch Time:", str((time.time()-et)/3600), 'hours',
          )

    if save_path is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            }, os.path.join(save_path, 'last.ckpt')
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--save_path", type=str, default="experiments")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1235)
    parser.add_argument("-v", "--voxel_size", type=float, default=None)
    parser.add_argument("-g", "--gpus", type=int, default=1)
    parser.add_argument("-a", "--augumentation", action="store_true")
    parser.add_argument("-r", "--resume_path", type=str, default=None)
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

    if torch.cuda.is_available():
        device = torch.device("cuda")

    train(save_path=save_path, run_name=args.run_name, augs=args.augumentation, resume_path=args.resume_path)
