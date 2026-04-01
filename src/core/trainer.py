from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy


class SegmentationTrainer:
    def __init__(self, config):
        self.config = config
        self._init_callbacks()
        self._init_logger()


    def _init_callbacks(self):
        self.best_miou = ModelCheckpoint(
            monitor='val/mIoU', 
            filename=f'best_model'+"-{epoch}-{val/mIoU}",
            save_top_k=1,
            mode='max',
            verbose=True,
        )

        self.last_checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            every_n_epochs=1,
            filename='last_model-{epoch}',
            verbose=False,
        )
    
        self.early_stop_callback = EarlyStopping(
            monitor='val/mIoU',
            patience=self.config.get('early_stop_patience', 10),
            mode='max',
            verbose=True,
        )

        self.lr_monitor = LearningRateMonitor(logging_interval='epoch')


    def _init_logger(self):
        self.logger = TensorBoardLogger(
            save_dir=self.config['log_dir'],
            name=self.config['experiment_name']
        )


    def fit(self, model, data_module):
        trainer = Trainer(
            max_epochs=self.config['epochs'],

            gpus=self.config.get('gpus', -1),
            sync_batchnorm=True if len(self.config['gpus']) != 1 else False,
            strategy=DDPStrategy(find_unused_parameters=False) if len(self.config['gpus']) != 1 else None,

            logger=self.logger,

            callbacks=[
                self.early_stop_callback,
                self.last_checkpoint_callback,
                self.best_miou,
                self.lr_monitor,
            ],

            precision=self.config['precision'],
            accumulate_grad_batches=self.config['accumulate_steps'],
            deterministic=False,
            resume_from_checkpoint=self.config['checkpoint_path'] if self.config.get('resume', False) else None,

        )

        trainer.fit(model, *data_module)
        return trainer