from pytorch_forecasting import QuantileLoss, TemporalFusionTransformer
from stock_TFT import TFTModel
import pandas as pd



def train_model():
    df = pd.read_csv("./data/Train万华化学.csv", sep=',')

    print(df.isna)

    df.insert(0, "time_idx", [i for i in range(df.shape[0])])
    df = df.drop(columns='date', axis=1)
    df = df.drop(columns='turn', axis=1)
    df = df.iloc[9:, ]
    df.insert(7, "next_close", df["close"].shift(-1))
    df_train = df.iloc[:-1, ]
    # print(df["next_close"].describe())
    print(df_train.columns)
    # print(df[-1, :])
    # df_train = df.iloc[:, :]
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

    # df_test = pd.read_csv("../auto-stock-trader/stockTrader/data/Test万华化学.csv", sep=',')
    
def eval(model: TemporalFusionTransformer):
    df_test = pd.read_csv("./data/Train万华化学.csv", sep=',')

    df_test = TFTModel.preprocess(df_test)

    model.predict(df_test)

    err = model.evaluate()

    print("MAE error: ", err)




if __name__ == "__main__":
    main()