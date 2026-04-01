from ..models.train_engine import SegmentationModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy


class SegmentationPredictor:    
    def __init__(self, checkpoint_path):
        self.model = self._load_model(checkpoint_path)
        self.config = self._load_config()
        self._init_logger()


    def _load_config(self):
        return self.model.hparams.config


    def _load_model(self, checkpoint_path):
        model = SegmentationModel.load_from_checkpoint(
            checkpoint_path,
        )
        return model


    def _init_logger(self):
        self.logger = TensorBoardLogger(
            save_dir=self.config['log_dir'],
            name=self.config['experiment_name']
        )


    def predict(self, test_loader):
        trainer = Trainer(
            gpus=self.config.get('gpus', -1),
            sync_batchnorm=True if len(self.config['gpus']) != 1 else False,
            strategy=DDPStrategy(find_unused_parameters=False) if len(self.config['gpus']) != 1 else None,

            logger=self.logger,
            precision=self.config['precision'],
            deterministic=False,
            accumulate_grad_batches=self.config['accumulate_steps'],
            detect_anomaly=False,
        )
        trainer.test(self.model, dataloaders=test_loader, verbose=True)
        
