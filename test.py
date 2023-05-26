import matplotlib.pyplot as plt
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import GroupNormalizer, QuantileLoss, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.utilities.types import (_EVALUATE_OUTPUT,
                                               _PREDICT_OUTPUT,
                                               EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS,
                                               LRSchedulerConfig)
from stock_TFT import TFTModel


def train_model():
    df = pd.read_csv("./data/Train万华化学.csv", sep=',')

    print(df.isna)

    df.insert(0, "time_idx", [i for i in range(df.shape[0])])
    df = df.drop(columns='date', axis=1)
    df = df.drop(columns='turn', axis=1)
    df = df.iloc[9:, ]
    df.insert(7, "next_close", df["close"].shift(-1))
    df_train = df.iloc[:-1, ]
    print(df_train.columns)
    df_train.dropna()
    df_test = df.iloc[544:, :]  
    
    model = TFTModel(
        df_train=df_train,
        max_encoder_length=24,
        max_prediction_length=6,
        batch_size=128
    )

    model.fit(
        max_epochs=30, 
        gradient_clip_val=0.1, 
        limt_train_batches=10, 
        learning_rate=0.01,
        hidden_size=64,
        attention_head_size=5,
        dropout=0.01,
        hidden_continuous_size=8,
        output_size=7,
        log_interval=10,
        reduce_on_plateau_patience=4
    )

    return model

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

    
def predict(file_path: str):
    df_test = pd.read_csv(file_path, sep=',')
    print("-------------processing data--------------")
    df_test = preprocess(df_test)
    print("-------------finish processing--------------")

    predict_length = 10
    final_predictions = []
    day = 0

    print("Start loading from model path.....")
    model = TemporalFusionTransformer.load_from_checkpoint("./lightning_logs/lightning_logs/version_3/checkpoints/epoch=29-step=300.ckpt")
    print("Finish loading from model path.....")

    while True:
        if day == predict_length:
            break
    
        model.eval()

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath='./lightning_logs/best/v1',
            filename='best-checkpoint',
            save_top_k=1,
            mode='min',
        )

        with torch.no_grad():
            predictions = model.predict(df_test)[0]
        
        final_predictions.extend(predictions)

        lr_logger = LearningRateMonitor()
        logger = TensorBoardLogger("lightning_logs")
        trainer = pl.Trainer(max_epochs=5, callbacks=[checkpoint_callback])

        left_index = 0 if day == 0 else day * len(predictions)

        train_data = df_test.iloc[left_index : left_index + len(predictions), :]

        # TODO: create a 
        train_dataset = create_dataset_from_df(train_data, 24, 6)

        trainer.fit(
            model=model,
            train_dataloaders=dataloader_from_dataset(train_dataset, True, 128, 0)
        )

        day += 1

        print(f"------------train {day}--------------")
        

    return final_predictions
    # # based on next_close value, determine action





if __name__ == "__main__":
    predicts = predict("./data/Test万华化学.csv")
    print(predicts)
