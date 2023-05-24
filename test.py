import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
from pytorch_forecasting import QuantileLoss, TemporalFusionTransformer
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

    
def predict():
    df_test = pd.read_csv("./data/Test万华化学.csv", sep=',')
    df_test = TFTModel.preprocess(df_test)

    predict_length = 20
    final_predictions = []
    day = 0

    model = TemporalFusionTransformer.load_from_checkpoint("./lightning_logs/lightning_logs/version_3/checkpoints/epoch=29-step=300.ckpt")
    
    while day <= predict_length:
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
        trainer = pl.Trainer(callbacks=[checkpoint_callback])

        left_index = 0 if day == 0 else day * len(predictions)

        train_data = df_test.iloc[left_index : left_index + len(predictions), :]
        train_dataset = TFTModel.create_dataset_from_df(train_data, 24, 6)

        # trainer.fit(
        #     model=model,
        #     train_dataloaders=TFTModel.dataloader_from_dataset(train_dataset, True, 128, 0)
        #     val_dataloaders=
        # )
        


    # based on next_close value, determine action





if __name__ == "__main__":
    predicts = TFTModel.predict("./data/Test万华化学.csv")
    print(predicts)
