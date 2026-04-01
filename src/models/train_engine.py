import pytorch_lightning as pl
import torch
from ..utils.metrics import SegmentationMetrics
import pandas as pd
import os
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import csv
import importlib
from collections import defaultdict
import torch.distributed as dist
from models.mssvt import MSSVT
import segmentation_models_pytorch as smp


class SegmentationModel(pl.LightningModule):    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters("config")
        self.ignore_index = config.get("ignore_index", None)
        self.config = config
        self.num_classes = config['num_classes']
        self.model = self._init_model()

        # Metrics
        self.train_metrics = SegmentationMetrics(config['num_classes'], ignore_index=self.ignore_index)
        self.val_metrics = SegmentationMetrics(config['num_classes'], ignore_index=self.ignore_index)
        self.test_metrics = SegmentationMetrics(config['num_classes'], ignore_index=self.ignore_index)
        
        # Loss Functions
        self.ce_loss = smp.losses.SoftCrossEntropyLoss(smooth_factor=0, ignore_index=self.ignore_index)
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True, ignore_index=self.ignore_index, smooth=0)
        # Combined Loss
        self.loss_fn = lambda logits, targets: 2.0 * self.ce_loss(logits, targets) + 5.0 * self.dice_loss(logits, targets)
        self.aux_loss_fn = lambda logits, targets: self.ce_loss(logits, targets)

        # Deep Supervision Settings
        self.deep_supervison = config['deep_supervision']
        self.aux_weights = config['aux_weights']


    def _init_model(self):
        return MSSVT(num_classes=self.num_classes)


    def forward(self, x):
        return self.model(x)


    def _shared_step(self, batch, stage):
        images, masks = batch
        if not self.deep_supervison:
            main_logits, aux_outputs = self.forward(images)
            main_loss = self.loss_fn(main_logits, masks)
            probs = main_logits.softmax(dim=1)
            preds = probs.argmax(dim=1)
            metrics = getattr(self, f"{stage}_metrics")
            metrics.add_batch(preds, masks)

            return {"loss": main_loss}
        else:
            main_logits, aux_outputs = self.forward(images)
            losses_dict: dict[str: float] = defaultdict(float)
            main_loss = self.loss_fn(main_logits, masks)
            losses_dict["main_loss"] = main_loss.item()
            aux_losses = 0
            if stage == "train":
                for name, logits_list in aux_outputs.items():
                    for i, logits in enumerate(logits_list):
                        aux_loss = self.aux_loss_fn(logits, masks)
                        losses_dict[f'aux-{name}-{i}'] = aux_loss.item()
                        aux_losses += aux_loss * self.aux_weights[i]

            total_loss = main_loss + aux_losses
            if stage == "train":
                losses_dict["loss"] = total_loss
            else:
                losses_dict["loss"] = total_loss.item()
            probs = main_logits.softmax(dim=1)
            preds = probs.argmax(dim=1)

            metrics = getattr(self, f"{stage}_metrics")
            metrics.add_batch(preds, masks)

            return losses_dict


    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "test")        
        return loss

    def _shared_epoch_end(self, outputs, stage):
        metrics = {}
        losses = [x['loss'] for x in outputs]

        if len(losses) > 0:
            if isinstance(losses[0], torch.Tensor):
                avg_loss = torch.stack(losses).mean()
            else:
                avg_loss = torch.tensor(losses).mean().to(self.device)
            metrics[f'{stage}/loss'] = avg_loss
        
        stage_metrics = getattr(self, f"{stage}_metrics")
        stage_metrics.synchronize_between_gpus()

        metrics[f'{stage}/acc'] = stage_metrics.compute_overall_accuracy()
        metrics[f'{stage}/mIoU'] = stage_metrics.compute_mean_iou()

        current_epoch = self.trainer.current_epoch    
        for key, value in metrics.items():        
            self.logger.experiment.add_scalar(f"manual/{key}", value, current_epoch)

        self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True)
        self._save_metrics_to_csv(stage, current_epoch, metrics)

        stage_metrics.reset()

    def training_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self.test_metrics.synchronize_between_gpus()
        metrics = {}
        metrics['mIoU'] = self.test_metrics.compute_mean_iou().numpy()
        metrics['OA'] = self.test_metrics.compute_overall_accuracy().numpy()
        metrics['Precision'] = self.test_metrics.compute_precision().numpy()
        metrics['Recall'] = self.test_metrics.compute_recall().numpy()
        metrics['F1'] = self.test_metrics.compute_f1score().numpy()
        metrics['IoU'] = self.test_metrics.compute_iou().numpy()

        self.test_metrics.plot_confusion_matrix(
            self.hparams.config["class_names"],
            os.path.join(self.logger.log_dir, "confusion_matrix.png")
        )

        self.save_report(metrics, output_path=self.logger.log_dir)
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=float(self.hparams.config['lr']),
            weight_decay=float(self.hparams.config.get('weight_decay', 1e-4))
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=95,
            eta_min=float(self.hparams.config['eta_min'])
        )
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.01,
            total_iters=5
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[5]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,   
            }
        }
    

    def save_report(self, metrics, output_path: str):
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            print("now rank is ", rank)
        else:
            rank = 0

        file_suffix = f"_gpu{rank}"

        classwise_data = []
        vertical_name = ["Precision", "Recall", "F1", "IoU"]
        for name in vertical_name:
            class_metircs = metrics[name]
            classwise_data.append(
                [name] + class_metircs.tolist()
            )

        horizontal_names = self.hparams.config["class_names"].copy()
        horizontal_names.insert(0, "Index")
        
        df_class = pd.DataFrame(
            classwise_data,
            columns=horizontal_names
        )

        df_class = df_class.round(4)

        df_overall = pd.DataFrame({
            "OA": [metrics["OA"].round(4)],
            "mIoU": [metrics["mIoU"].round(4)]
        })
        
        df_class.to_csv(os.path.join(output_path, f"class{file_suffix}.csv"), index=False)
        df_overall.to_csv(os.path.join(output_path, f"overall{file_suffix}.csv"), index=False)

    
    def _save_metrics_to_csv(self, stage: str, epoch: int, metrics):
        filename = f"{stage}_metrics.csv"
        stage_metrics = getattr(self, f"{stage}_metrics")
        metrics[f'{stage}/Class_IoU'] = stage_metrics.compute_iou().numpy()    

        log_dir = (
            self.logger.log_dir if hasattr(self.logger, 'log_dir') 
            else self.trainer.default_root_dir
        )
        csv_path = os.path.join(log_dir, filename)
        
        class_names = self.hparams.config['class_names']
        columns = ['epoch', 'loss', 'acc', 'mIoU']
        columns.extend(class_names)
        row_data = {
            'epoch': epoch,
            'loss': metrics[f'{stage}/loss'].item(),
            'acc': metrics[f'{stage}/acc'].item(),
            'mIoU': metrics[f'{stage}/mIoU'].item(),
            }
        for i, class_name in enumerate(class_names):
            row_data[class_name] = metrics[f'{stage}/Class_IoU'][i]
        
        file_exists = os.path.exists(csv_path)
    
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
    
