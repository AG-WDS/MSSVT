import sys
sys.path.append('./')
import argparse
import yaml
from src.data.datasets import VRLDatset
from src.data.transforms import get_VRLDataset_transforms
from src.models.train_engine import SegmentationModel
from src.core.trainer import SegmentationTrainer
from torch.utils.data import DataLoader
import os
import pytorch_lightning as pl


def run_training(config):
    """
    Runs the training process based on the provided configuration dictionary.

    Args:
        config (dict): Dictionary containing the training configuration.

    Returns:
        pytorch_lightning.Trainer: The trainer instance after fitting.
                                   Returns None if an error occurs before trainer initialization.
    """
    # Set random seed
    seed = config['random_seed']
    pl.seed_everything(seed, workers=True)

    print(f"Running training for experiment: {config.get('experiment_name', 'N/A')}")
    print(f"Config: {config}") # Print config for debugging

    # Initialize data
    train_transform = get_VRLDataset_transforms('train')
    val_transform = get_VRLDataset_transforms('val')

    dataset_dir = config['dataset_dir']
    train_images_dir = os.path.join(dataset_dir, config['train_images_dir'])
    train_masks_dir = os.path.join(dataset_dir, config['train_masks_dir'])
    val_images_dir = os.path.join(dataset_dir, config['val_images_dir'])
    val_masks_dir = os.path.join(dataset_dir, config['val_masks_dir'])
    
    train_dataset = VRLDatset(
        train_images_dir,
        train_masks_dir,
        transform=train_transform,
    )

    val_dataset = VRLDatset(
        val_images_dir,
        val_masks_dir,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )
    print("Data loaded successfully.")

    # Initialize model
    if config.get('resume', False):
        checkpoint_path = config.get('checkpoint_path')
        if not checkpoint_path or not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Resume=True but checkpoint_path missing or not found: {checkpoint_path}")
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = SegmentationModel.load_from_checkpoint(checkpoint_path)
    else:
        print("Initializing model from scratch.")
        model = SegmentationModel(config)

    print("Model initialized successfully.")

    segmentation_trainer_instance = None    
    print("Initializing SegmentationTrainer...")
    segmentation_trainer_instance = SegmentationTrainer(config)

    print("Starting trainer.fit()...")
    trainer = segmentation_trainer_instance.fit(model, [train_loader, val_loader])
    print("trainer.fit() finished.")

    return trainer # Return the pytorch_lightning.Trainer instance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/default.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"Running train.py as a standalone script with config: {args.config}")

    trainer_instance = run_training(config)
    if trainer_instance:
        print("\nStandalone training completed.")
        if hasattr(trainer_instance, 'checkpoint_callback') and trainer_instance.checkpoint_callback:
            print(f"Best checkpoint path: {trainer_instance.checkpoint_callback.best_model_path}")
            print(f"Best validation score: {trainer_instance.checkpoint_callback.best_model_score}")
        if hasattr(trainer_instance, 'logger') and hasattr(trainer_instance.logger, 'log_dir'):
                print(f"Log Directory: {trainer_instance.logger.log_dir}")
