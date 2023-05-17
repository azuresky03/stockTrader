import akshare as ak
import pandas as pd
import itertools
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl import config_tickers

START_DATE = "2014-01-01"  # "2014-01-06", should contain "-"
END_DATE = "2023-05-01"  # "2021-10-01", should contain "-"
TIME_INTERVAL = "daily"  # choice of {"daily", "1", "5", "15", "30", "60"}
ADJUST = "hfq"  # choice of {'', 'qfq', 'hfq'}

tickers = [
    "600519",
    "000858",
    "601888",
    "002594",
    "002230",
    "000725",
    "601398",
    "601939",
    "601628",
    "600276",
    "300015",
    "600436",
    "601899",
    "600309",
    "601012",
    "600887",
    "002714",
    "000895",
    "601857",
    "600028",
    "601088",
    "601668",
    "601390",
    "000002",
    "002352",
    "601766",
    "600029",
    "600941",
    "601728",
    "600900",
]

INDICATORS = [
    "macd",
    "boll",
    "rsi",
    "ppo",
    "log-ret",
    "vwma",
    "adx",
    "mfi",
]


def download_data(
    tickers=tickers,
    start_date=START_DATE,
    end_date=END_DATE,
    time_interval=TIME_INTERVAL,
    adjust=ADJUST,
):
    """
    Download data from akshare
    :param tickers: (list) list of stock tickers
    :param start_date: (str) start date
    :param end_date: (str) end date
    :param time_interval: (str) time interval
    :param adjust: (str) adjust method
    :return: (df) pandas dataframe
    """
    df = pd.DataFrame()

    # change time format
    start_date = start_date.replace("-", "")
    end_date = end_date.replace("-", "")
    for ticker in tickers:
        try:
            temp_data = ak.stock_zh_a_hist(
                symbol=ticker,
                period=time_interval,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            temp_data["ticker"] = ticker
            df = pd.concat([df, temp_data])
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    # store data into a csv file
    df.to_csv("raw_data.csv", index=False)
    return df


def downlaod_data_yahoo():
    df = YahooDownloader(
        start_date=START_DATE,
        end_date=END_DATE,
        ticker_list=config_tickers.DOW_30_TICKER,
    ).fetch_data()

    df.to_csv("raw_dow.csv", index=False)

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(
        pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
    )
    combination = list(itertools.product(list_date, list_ticker))
    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        processed, on=["date", "tic"], how="left"
    )
    processed_full = processed_full[processed_full["date"].isin(processed["date"])]
    processed_full = processed_full.sort_values(["date", "tic"])
    processed_full = processed_full.fillna(0)
    processed_full.sort_values(["date", "tic"], ignore_index=True)
    processed_full.to_csv("dow_30.csv", index=False)


def download_data_crypto():
    # list_name = ak.crypto_name_url_table()
    # list_name = list_name["name"].tolist()
    # df = pd.DataFrame()
    # for symbol in list_name:
    #     crypto_hist_df = ak.crypto_hist(
    #         symbol="BTC", period="每日", start_date="20151020", end_date="20230501"
    #     )
    #     crypto_hist_df["tic"] = symbol
    #     df = pd.concat([df, crypto_hist_df])
    # df.to_csv("raw_crypto.csv", index=False)
    crypto_hist_df = ak.crypto_hist(symbol="BTC", period="每日", start_date="20151020", end_date="20201023")
    crypto_hist_df["tic"] = "BTC"
    crypto_hist_df.to_csv("raw_crypto.csv", index=False)


def clean_data(data):
    df = data.copy()
    df = df.sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]
    merged_closes = df.pivot_table(index="date", columns="tic", values="close")

    # Fill missing close prices using the last valid close price
    filled_merged_closes = merged_closes.fillna(method='ffill')

    # Fill open, high, low with filled close prices
    filled_opens = df.pivot_table(index="date", columns="tic", values="open").fillna(filled_merged_closes)
    filled_highs = df.pivot_table(index="date", columns="tic", values="high").fillna(filled_merged_closes)
    filled_lows = df.pivot_table(index="date", columns="tic", values="low").fillna(filled_merged_closes)

    # Fill volume with 0
    filled_volumes = df.pivot_table(index="date", columns="tic", values="volume").fillna(0)

    # Merge filled data back into the original DataFrame
    df_filled = pd.concat([filled_opens.stack(), filled_highs.stack(), filled_lows.stack(),
                           filled_merged_closes.stack(), filled_volumes.stack()], axis=1)
    df_filled = df_filled.reset_index()
    df_filled.columns = ["date", "tic", "open", "high", "low", "close", "volume"]

    return df_filled



def preprocess_data(df: pd.DataFrame):
    column_mapping = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "换手率": "turnover",
        "ticker": "tic",
    }
    column_drop = ["change", "turnover", "pct_chg", "amplitude", "amount"]
    df["ticker"] = df["ticker"].astype(str)
    df["ticker"] = df["ticker"].apply(lambda x: x.zfill(6))

    df = df.rename(columns=column_mapping)
    df = df.sort_values(by=["date", "tic"]).drop(columns=column_drop)
    # move tic column to the second column
    tic_column = df.pop("tic")
    df.insert(1, "tic", tic_column)

    df = clean_data(df)

    # add technical indicators
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)
    processed = processed.sort_values(["date", "tic"], ignore_index=True)
    processed.to_csv("processed_data.csv", index=False)
    return processed


if __name__ == "__main__":
    # crypto_js_spot_df = ak.crypto_js_spot()
    # print(crypto_js_spot_df)
    raw_df = download_data()
    # raw_df = pd.read_csv("raw_data.csv")
    print("raw_data:", raw_df.shape)
    processed_df = preprocess_data(raw_df)
    print("processed_data:", processed_df.shape)
