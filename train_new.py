import argparse
import os
from datetime import datetime
import time

import gin
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
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
    
    save_path = os.path.join(
        save_path,
        gin.query_parameter("train.model_name") + '_' + str(gin.query_parameter("DimensionlessCoordinates.voxel_size"))[2:]
    )
        
    train_batch_size = gin.query_parameter("ScanNetRGBDataModule.train_batch_size")
        
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=args.ddp_backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = torch.device("cuda")
    total_batch_size = ddp_world_size * train_batch_size
    print("DDP world size:", ddp_world_size, "Total batch size:", total_batch_size)
    
    if args.voxel_size is not None:
        gin.bind_parameter("DimensionlessCoordinates.voxel_size", args.voxel_size)
    if ddp_world_size > 1:
        gin.bind_parameter("train.gpus", ddp_world_size)
        gin.bind_parameter("SegmentationTrainer.lr", gin.query_parameter("SegmentationTrainer.lr")*ddp_world_size)

    torch.manual_seed(1337 + seed_offset)
    
    now = datetime.now().strftime('%m-%d-%H-%M-%S')
    run_name = run_name + "_" + now
    # Only create save path on master process
    if master_process:
        ensure_dir(save_path)

    ## get data and model
    data_module = get_data_module(data_module_name)()
    model = get_model(model_name)()
    model = model.to(device)

    if ddp_world_size > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    hparams = logged_hparams()

    # compile the model (needs PyTorch 2.0 or higher)
    if args.compile:
        print("compiling the model... (takes a minute)")
        unoptimized_model = model
        model = torch.compile(model)

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

    # write config file
    if master_process:
        with open(os.path.join(save_path, "config.gin"), "w") as f:
            f.write(gin.operative_config_str())

    data_module.setup()
    # gin.bind_parameter("DimensionlessCoordinates.voxel_size", 0.1)
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

    if augs:
        resolutions = [0.02, 0.05, 0.10]
        max_step = len(resolutions) * max_step
    else:
        resolutions = None

    max_epoch = max_epoch if max_epoch != 0 else int(max_step/(len(train_dataloader)/train_batch_size))
    trainer.max_epoch = max_epoch
    for epoch in range(init_epoch, max_epoch):
        # # train for one epoch
        # if augs and resolutions is not None:
        #     train_dataloader.dataset.transform.transforms[0].voxel_size = resolutions[epoch % 3]
        #     print(train_dataloader.dataset.transform.transforms[0].voxel_size)
        train_one_epoch(hparams, trainer, train_dataloader, epoch + init_epoch, max_epoch, save_path, device, master_process, augs, resolutions)

        if check_val_every_n_epoch != 0 and (epoch % check_val_every_n_epoch == 0):
            val_one_epoch(hparams, trainer, val_dataloader, epoch + init_epoch, save_path, device, master_process)
            
    if ddp:
        destroy_process_group()


def val_one_epoch(hparams, trainer, dataloader, epoch, save_path, device, master_process):
    mloss = 0
    for i, batch in enumerate(dataloader):
        # if i > 30:
        #     break
        # move data to the same device as model
        loss = trainer.val_one_step(batch)

        if master_process:
            print("Iterations:", str(i),
                "Train Loss", loss.item(),
                )

        mloss += loss.item()
        del loss

    if master_process:
        print("Epoch val loss:", str(mloss/(i+1)))
        trainer.on_validation_epoch_end(epoch, master_process, save_path)

def train_one_epoch(hparams, trainer, train_dataloader, epoch, max_epoch, save_path, device, master_process, augs=False, resolutions=None):
    mloss = 0
    et = time.time()

    # # move data to the same device as model
    
    # print()
    for i, batch in enumerate(train_dataloader):
        t0 = time.time()
        loss = trainer.train_one_step(batch)

        if master_process:
            print("Iterations / Epochs / Total Epochs:", str(i), "/", str(epoch), "/", str(max_epoch),
                "Train Loss:", loss.item(), 
                "Batch Size:", batch["batch_size"], 
                "Batch Time:", str(time.time()-t0), 'seconds',
                )

        mloss += loss.item()
        del loss

    if master_process:
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
    parser.add_argument("-a", "--augumentation", action="store_true")
    parser.add_argument("-r", "--resume_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--compile", action="store_true", help="use compile in PyTorch 2.0.")
    
    # ddp settings
    parser.add_argument("--ddp_backend", type=str, default="nccl", help="'nccl', 'gloo', etc.")
    args = parser.parse_args()

    gin.parse_config_file(args.config)

    train(save_path=args.save_path, run_name=args.run_name, augs=args.augumentation, resume_path=args.resume_path)
