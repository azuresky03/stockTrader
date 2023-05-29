from pytorch_forecasting import SMAPE
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


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df.insert(0, "time_idx", [i for i in range(df.shape[0])])
    df = df.drop(columns='date', axis=1)
    df = df.drop(columns='turn', axis=1)
    df.insert(7, "next_close", df["close"].shift(-1))
    df_clean = df.iloc[:-1, ]
    return df_clean

def create_dataset_from_df(data: pd.DataFrame, max_encoder_length: int, max_prediction_length: int):
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


def dataloader_from_dataset(dataset: TimeSeriesDataSet, isTrain: bool, batch_size: int, num_workers: int = 0):
    dataloader = dataset.to_dataloader(
        train=isTrain, batch_size=batch_size, num_workers=num_workers
    )

    return dataloader

def train():
    # hyper parameters
    max_epochs = 30
    gradient_clip_val = 0.1
    limt_train_batches = 10
    learning_rate = 0.03
    hidden_size = 64
    attention_head_size = 4
    dropout = 0.2
    lstm_layers = 2
    hidden_continuous_size = 18
    output_size = 7
    log_interval = 10
    reduce_on_plateau_patience = 4


    df_train = pd.read_csv("./data/Train万华化学.csv", sep=',')

    df_train = preprocess(df_train)
    # print(df_train["next_close"].describe())

    df_test = pd.read_csv("./data/Test万华化学.csv", sep=",")
    df_test = preprocess(df_test)
    # print(df_test.shape)
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./lightning_logs/best/v1',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        # gpus=[0],  # train on CPU, use gpus = [0] to run on GPU
        gradient_clip_val=gradient_clip_val,
        # early_stop_callback=early_stop_callback,
        limit_train_batches=limt_train_batches,  # running validation every 30 batches
        # fast_dev_run=True,  # comment in to quickly check for bugs
        callbacks=[lr_logger],
        logger=logger,
    )


    train_dataset = create_dataset_from_df(df_train, 24, df_test.shape[0])


    tft = TemporalFusionTransformer.from_dataset(
        dataset=train_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        lstm_layers=lstm_layers,
        hidden_continuous_size=hidden_continuous_size,
        output_size=output_size,  #  7 quantiles by default
        loss=QuantileLoss(),
        # loss=SMAPE(),
        log_interval=log_interval,
        reduce_on_plateau_patience=reduce_on_plateau_patience,
    )

    validation = TimeSeriesDataSet.from_dataset(train_dataset, df_train, predict=True, stop_randomization=True)

    trainer.fit(
        tft,
        train_dataloaders=dataloader_from_dataset(train_dataset, isTrain=True, batch_size=128),
        val_dataloaders=dataloader_from_dataset(validation, isTrain=False, batch_size=1280)
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    print("best model path: ", best_model_path)
    return


def predict(file_path: str):
    df_test = pd.read_csv(file_path, sep=",")
    df_test = preprocess(df_test)

    model = TemporalFusionTransformer.load_from_checkpoint("./lightning_logs/lightning_logs/version_2/checkpoints/epoch=29-step=300.ckpt")
    model.eval()

    with torch.no_grad():
        predictions = model.predict(df_test, return_x=True)

    # predictions_vs_actuals = model.calculate_prediction_actual_by_variable(x, predictions)
    # all_features = list(set(predictions_vs_actuals['support'].keys())-set(['num_sold_lagged_by_365', 'num_sold_lagged_by_30', 'num_sold_lagged_by_7']))
    # for feature in all_features:
    #     model.plot_prediction_actual_by_variable(predictions_vs_actuals, name=feature)

    return predictions



if __name__ == "__main__":
    train()
    # predictions = predict("./data/Test万华化学.csv")
    # print(predictions)


    # best model path: lightning_logs/lightning_logs/version_0/checkpoints/epoch=29-step=300.ckpt
    # increased dropout: lightning_logs/lightning_logs/version_2/checkpoints/epoch=29-step=300.ckpt