import gin
import numpy as np
import torch
import torchmetrics
import MinkowskiEngine as ME
from src.utils.metric import per_class_iou
import os

from torch.optim.lr_scheduler import CosineAnnealingLR

@gin.configurable
class SegmentationTrainer(object):
    def __init__(
        self,
        model,
        num_classes,
        lr,
        momentum,
        weight_decay,
        warmup_steps_ratio,
        max_steps,
        max_epoch,
        best_metric_type,
        device,
        ignore_label=255,
        dist_sync_metric=False,
        lr_eta_min=0.,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        self.model = model
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr = lr
        self.max_steps = max_steps
        self.max_epoch = max_epoch
        self.lr_eta_min = lr_eta_min
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
        if torch.cuda.is_available():
            self.criterion = self.criterion.to(self.device)
        self.best_metric_value = -np.inf if best_metric_type == "maximize" else np.inf
        self.metric = torchmetrics.ConfusionMatrix(
            num_classes=num_classes,
            compute_on_step=False,
            dist_sync_on_step=dist_sync_metric
        )

    def configure_optimizer_lrscheduler(self):
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_steps,
            eta_min=self.lr_eta_min,
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "step"
            }
        }

    def train_one_step(self, batch):
        """Do forward, backward and parameter update."""
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        # move data to CUDA
        if torch.cuda.is_available():
            features=batch["features"].to(self.device)
            coordinates=batch["coordinates"].to(self.device)
            labels=batch["labels"].to(self.device)

        input_data = ME.TensorField(
            features=features,
            coordinates=coordinates,
            quantization_mode=self.model.QMODE
        )

        logits = self.model(input_data)
        loss = self.criterion(logits, labels)

        # backward and parameter update
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

    def val_one_step(self, batch):
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()
            self.zero_grad()

            # move data to CUDA
            if torch.cuda.is_available():
                features=batch["features"].to(self.device)
                coordinates=batch["coordinates"].to(self.device)
                labels=batch["labels"].to(self.device)

            input_data = ME.TensorField(
                features=features,
                coordinates=coordinates,
                quantization_mode=self.model.QMODE
            )

            logits = self.model(input_data)
            loss = self.criterion(logits, labels)
            pred = logits.argmax(dim=1, keepdim=False)
            mask = labels != self.ignore_label
            self.metric(pred[mask].detach().cpu(), labels[mask].detach().cpu())
            torch.cuda.empty_cache()

        return loss

    def on_validation_epoch_end(self, epoch, save_path=None):
        confusion_matrix = self.metric.compute().cpu().numpy()
        self.metric.reset()
        ious = per_class_iou(confusion_matrix) * 100
        accs = confusion_matrix.diagonal() / confusion_matrix.sum(1) * 100
        miou = np.nanmean(ious)
        macc = np.nanmean(accs)

        def compare(prev, cur):
            return prev < cur if self.best_metric_type == "maximize" else prev > cur
        
        if compare(self.best_metric_value, miou):
            self.best_metric_value = miou
            if save_path is not None:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, os.path.join(save_path, 'best_epoch_' + str(epoch) + '_iter_' + str(int(self.max_steps*epoch/self.max_epoch)) + '.ckpt')
                )

        print("val_best_mIoU:", self.best_metric_value,
              "val_mIoU", miou,
              "val_mAcc", macc,
              )

    def zero_grad(self):
        self.optimizer.zero_grad()
