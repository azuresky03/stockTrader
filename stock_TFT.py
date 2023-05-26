import pytorch_forecasting
from pytorch_forecasting.models.base_model import Prediction
import torch
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer


class TFTModel():

    @classmethod
    def create_dataset_from_df(cls, data: pd.DataFrame, max_encoder_length: int, max_prediction_length: int):
        return TimeSeriesDataSet(
            data=data,
            time_idx="time_idx",
            target="next_close",
            group_ids=["code"],               # only one time series for now
            min_encoder_length=0,
            min_prediction_length=1,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=["code"],
            # ["code", , "tradestatus", "adjustflag"],       FIXME: how to solve this
            static_reals=[],
            time_varying_unknown_categoricals=[],
            time_varying_known_categoricals=[],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["open", "high", "low", "close",
                                        "preclose", "volume", "amount", "pctChg", "next_close"],
            variable_groups={},
            target_normalizer=GroupNormalizer(
                groups=["code"],
                # transformation=""
            ),
            add_relative_time_idx=True,  # add as feature
            add_target_scales=True,  # add as feature
            add_encoder_length=True,  # add as feature
        )
    

    @classmethod
    def preprocess(cls, df: pd.DataFrame) -> pd.DataFrame:
        df.insert(0, "time_idx", [i for i in range(df.shape[0])])
        df = df.drop(columns='date', axis=1)
        df = df.drop(columns='turn', axis=1)
        df.insert(7, "next_close", df["close"].shift(-1))
        df_clean = df.iloc[:-1, ]
        return df_clean
    

    @classmethod
    def dataloader_from_dataset(cls, dataset: TimeSeriesDataSet, isTrain: bool, batch_size: int, num_workers: int = 0):
        dataloader = dataset.to_dataloader(
            train=isTrain, batch_size=batch_size, num_workers=num_workers
        )

        return dataloader
     

    def __init__(self, df_train: pd.DataFrame, max_prediction_length: int, max_encoder_length: int, batch_size: int):
        self.bes_model_path = ""
        self.batch_size = batch_size

        # self.train_dataset = TFTModel.create_dataset_from_df(df_train)

        # self.validation_dataset = TimeSeriesDataSet.from_dataset(
        #     self.train_dataset, df_train, predict=True, stop_randomization=True
        # )

        # self.train_dataloader = self.train_dataset.to_dataloader(
        #     train=True, batch_size=batch_size, num_workers=0
        # )

        # self.val_dataloader = self.validation_dataset.to_dataloader(
        #     train=False, batch_size=batch_size * 10, num_workers=0
        # )

    

    def fit(self, 
        max_epochs: int, 
        gradient_clip_val: int, 
        limt_train_batches: int, 
        learning_rate: int,
        hidden_size: int,
        attention_head_size: int,
        dropout: int,
        hidden_continuous_size: int,
        output_size: int,
        log_interval: int,
        reduce_on_plateau_patience: int
    ):
        # early_stop_callback = EarlyStopping(
        #     monitor="val_loss",
        #     min_delta=1e-4,
        #     patience=10,
        #     verbose=False,
        #     mode="min"
        # )

        lr_logger = LearningRateMonitor()
        logger = TensorBoardLogger("lightning_logs")
        # create trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            # gpus=[0],  # train on CPU, use gpus = [0] to run on GPU
            gradient_clip_val=gradient_clip_val,
            # early_stop_callback=early_stop_callback,
            limit_train_batches=limt_train_batches,  # running validation every 30 batches
            # fast_dev_run=True,  # comment in to quickly check for bugs
            callbacks=[lr_logger],
            logger=logger,
        )

        tft = TemporalFusionTransformer.from_dataset(
            self.train_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            output_size=output_size,  #  7 quantiles by default
            loss=QuantileLoss(),
            log_interval=log_interval,
            reduce_on_plateau_patience=reduce_on_plateau_patience,
        )
        
        self.trainer.fit(
            tft,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader
        )

        self.bes_model_path = self.trainer.checkpoint_callback.best_model_path
        print(f"Store trained model into path: lightning_logs/version_{self.trainer.logger.version}/epoch_{self.trainer.current_epoch}.ckpt")

        
    @classmethod
    def predict(self, test_file_path: str):
        df_test = pd.read_csv(test_file_path, sep=',')
        df_test = TFTModel.preprocess(df_test)

        predict_length = 10
        final_predictions = []
        day = 0

        model = TemporalFusionTransformer.load_from_checkpoint("./lightning_logs/lightning_logs/version_3/checkpoints/epoch=29-step=300.ckpt")
        
        while day <= predict_length:
            # if day != 0:
            #     model = model.load_from_checkpoint(checkpoint_path="./lightning_logs/best/v1/best-checkpoint.ckpt")

            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath='./lightning_logs/best/v1',
                filename='best-checkpoint',
                save_top_k=1,
                mode='min',
            )

            predictions = model.predict(df_test)[0]
            final_predictions.extend(predictions)

            lr_logger = LearningRateMonitor()
            logger = TensorBoardLogger("lightning_logs")
            trainer = pl.Trainer(max_epochs=30, callbacks=[])

            left_index = 0 if day == 0 else day * len(predictions)

            train_data = df_test.iloc[left_index : left_index + len(predictions), :]
            if train_data.shape[0] == 0:
                print("read ")

            train_dataset = TFTModel.create_dataset_from_df(train_data, 24, 6)

            trainer.fit(
                model=model,
                train_dataloaders=TFTModel.dataloader_from_dataset(train_dataset, True, 128, 0)
            )

            # save model
            trainer.checkpoint_callback
            day += 1

        return final_predictions

    

    def evaluate(self, predictions: np.ndarray, actuals: np.ndarray):
        diff = predictions - actuals
        mae = np.sum(np.absolute(diff))
        return mae

