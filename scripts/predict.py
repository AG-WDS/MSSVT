import sys
sys.path.append('./')
import argparse
import yaml
from src.data.datasets import VRLDatset
from src.data.transforms import get_VRLDataset_transforms
from src.core.predictor import SegmentationPredictor
from torch.utils.data import DataLoader
import os
import pytorch_lightning as pl

def run_prediction(config, checkpoint_path):
    """
    Runs the prediction process using a specific checkpoint and configuration.

    Args:
        config (dict): Dictionary containing the overall configuration (including dataset paths).
        checkpoint_path (str): Path to the model checkpoint file (.ckpt).

    Returns:
        str: The path to the directory where prediction results (overall.csv, class.csv) are saved.
             Returns None if an error occurs before predictor initialization or logging dir is inaccessible.
    """
    seed = config['random_seed']
    pl.seed_everything(seed, workers=True)

    print(f"Running prediction using checkpoint: {checkpoint_path}")
    print(f"Using config (primarily for dataset paths): {config}")


    predictor = None
    predictor_log_dir = None

    predictor = SegmentationPredictor(checkpoint_path)
    print("Predictor initialized successfully.")

    test_transform = get_VRLDataset_transforms('test')

    dataset_dir = config['dataset_dir']
    test_images_dir = os.path.join(dataset_dir, config['test_images_dir'])
    test_masks_dir = os.path.join(dataset_dir, config['test_masks_dir'])

    test_dataset = VRLDatset(
        test_images_dir,
        test_masks_dir,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    print("Test data loaded successfully.")

    # Run prediction
    print("Starting predictor.predict()...")
    predictor.predict(test_loader)
    print("predictor.predict() finished.")

    if hasattr(predictor, 'logger') and hasattr(predictor.logger, 'log_dir'):
        predictor_log_dir = predictor.logger.log_dir
        print(f"Prediction results log dir: {predictor_log_dir}")
    else:
        print("Warning: Could not get prediction log directory from predictor.")
        pass

    return predictor_log_dir # Return the log directory path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/default.yaml", help="Path to the base YAML configuration file (used for dataset paths, batch size, etc.).")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint file (.ckpt) for prediction.")
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        experiment_config = yaml.safe_load(f)

    prediction_log_dir = run_prediction(experiment_config, args.checkpoint)
    if prediction_log_dir:
        print(f"\nStandalone prediction completed. Results in: {prediction_log_dir}")

