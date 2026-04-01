import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import torch.distributed as dist


class SegmentationMetrics:    
    def __init__(self, num_classes, device='cuda', ignore_index=None):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
        self.ignore_index = ignore_index


    def update(self, pred, target):
        device = pred.device
        if self.confusion_matrix.device != device:
            self.confusion_matrix = self.confusion_matrix.to(device)

        # Ensure tensors are on the correct device and flatten
        pred = pred.flatten()
        target = target.flatten()

        # Create a mask for valid pixels (not ignore_index and within class bounds)
        valid_mask = (target != self.ignore_index) if self.ignore_index is not None else torch.ones_like(target, dtype=torch.bool)
        valid_mask &= (target >= 0) & (target < self.num_classes) # Also ensure target is within bounds

        # Apply the mask
        pred_valid = pred[valid_mask]
        target_valid = target[valid_mask]

        # Calculate confusion matrix entries only for valid pixels
        # Note: Need target_valid to be long for bincount index
        labels = self.num_classes * target_valid.long() + pred_valid.long()
        counts = torch.bincount(labels, minlength=self.num_classes**2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)


    def add_batch(self, preds, targets):
        self.update(preds, targets)


    def synchronize_between_gpus(self):
        if dist.is_available() and dist.is_initialized():
            dist.barrier() 
            dist.all_reduce(self.confusion_matrix, op=dist.ReduceOp.SUM)    
    
    
    def _compute_safe_division(self, numerator, denominator):
        denominator = denominator.float()
        denominator[denominator == 0] = 1e-8
        result = numerator.float() / denominator
        return result.cpu()


    def compute_precision(self):
        tp = torch.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(0) - tp # Sum of column - diagonal
        return self._compute_safe_division(tp, tp + fp)


    def compute_recall(self):
        tp = torch.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(1) - tp # Sum of row - diagonal
        return self._compute_safe_division(tp, tp + fn)
    

    def compute_f1score(self):
        precision = self.compute_precision() # Move back to device for calc
        recall = self.compute_recall()
        # Ensure precision + recall is float for division
        denominator = (precision + recall).float()
        denominator[denominator == 0] = 1e-8 # Avoid NaN
        f1 = 2 * precision * recall / denominator
        return f1.cpu() # Return on CPU


    def compute_iou(self):
        intersection = torch.diag(self.confusion_matrix) # TP
        union = self.confusion_matrix.sum(0) + self.confusion_matrix.sum(1) - intersection # TP+FP + TP+FN - TP = TP+FP+FN
        return self._compute_safe_division(intersection, union)


    def compute_overall_accuracy(self):
        correct = torch.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return (correct.float() / total.float()).cpu()


    def compute_mean_iou(self, ignore_background=True):  # Option to ignore background class (index 0) in mIoU
        iou = self.compute_iou()
        # Remove NaN values which can occur if a class had zero union
        iou = iou[~torch.isnan(iou)]
        if ignore_background and self.num_classes > 1 and len(iou) > 0:
             valid_iou = []
             for i in range(self.num_classes):
                 if i != self.ignore_index: # Explicitly skip ignore_index
                    tp = self.confusion_matrix[i, i]
                    union = self.confusion_matrix[i, :].sum() + self.confusion_matrix[:, i].sum() - tp
                    if union > 0:
                        valid_iou.append(tp.float() / union.float())
             if not valid_iou:
                 return torch.tensor(float('nan'))
             return torch.stack(valid_iou).mean().cpu()
        elif len(iou) == 0:
             return torch.tensor(float('nan')) # Or 0.0
        else:
             return iou.mean().cpu()


    def reset(self):
        self.confusion_matrix.zero_()


    def plot_confusion_matrix(self, class_names, save_path=None, normalize='true'):
        """Plots confusion matrix. normalize can be 'true', 'pred', or None."""
        if self.confusion_matrix.sum() == 0:
             print("Warning: Confusion matrix is empty, skipping plot.")
             return
        
        cm = self.confusion_matrix.cpu().numpy()

        row_sums = cm.sum(axis=1)[:, np.newaxis]
        # Avoid division by zero for rows with no samples
        row_sums[row_sums == 0] = 1
        cm_norm = cm.astype('float') / row_sums
        title = "Normalized Confusion Matrix"

        if len(class_names) != self.num_classes:
             print(f"Warning: Number of class names ({len(class_names)}) does not match num_classes ({self.num_classes}). Using indices.")
             plot_class_names = [str(i) for i in range(self.num_classes)]
        else:
             plot_class_names = class_names

        plt.figure(figsize=(8, 8))
        plt.imshow(cm_norm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(plot_class_names))
        plt.xticks(tick_marks, plot_class_names, rotation=45, ha='center')
        plt.yticks(tick_marks, plot_class_names)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.grid(False) # Turn off grid lines

        # Print values in cells
        fmt = '.2f'
        thresh = cm_norm.max() / 2.

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                # Only display text if value is meaningful
                value = cm_norm[i, j]
                if value > 1e-4:
                    plt.text(j, i, format(value * 100, fmt) + '%',
                             ha="center", va="center",
                             color="white" if cm_norm[i, j] > thresh else "black")

        plt.tight_layout() # Adjust layout

        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        plt.close()

