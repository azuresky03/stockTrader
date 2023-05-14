import akshare as ak
import pandas as pd
import itertools
from stockstats import StockDataFrame as Sdf
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

START_DATE = "2014-06-01"  # "2014-01-06", should contain "-"
END_DATE = "2023-04-30"  # "2021-10-01", should contain "-"
TIME_INTERVAL = "daily"  # choice of {"daily", "1", "5", "15", "30", "60"}
ADJUST = "hfq"  # choice of {'', 'qfq', 'hfq'}

tickers = [
    "sh600519",
    "sz000858",
    "sh601888",
    "sz002594",
    "sz002230",
    "sz000725",
    "sh601398",
    "sh601939",
    "sh601628",
    "sh600276",
    "sz300015",
    "sh600436",
    "sh601899",
    "sh600309",
    "sh601012",
    "sh600887",
    "sz002714",
    "sz000895",
    "sh601857",
    "sh600028",
    "sh601088",
    "sh601668",
    "sh601390",
    "sz000002",
    "sz002352",
    "sh601766",
    "sh600029",
    "sh600941",
    "sh601728",
    "sh600900",
]

INDICATORS = [
    "macd",
    "adx",
    "rsi_30",
    "boll_ub",
    "boll_lb",
    "close_30_sma",
    "close_60_sma",
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

    if TIME_INTERVAL == "daily":
        # change time format
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")
        # remove front 2 letters for sh and sz
        new_tickers = [ticker[2:] for ticker in tickers]
        for ticker in new_tickers:
            try:
                temp_data = ak.stock_zh_a_hist(
                    symbol=ticker,
                    period=time_interval,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )
                temp_data["ticker"] = tickers[new_tickers.index(ticker)]
                df = pd.concat([df, temp_data])
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
    elif TIME_INTERVAL in {"1", "5", "15", "30", "60"}:
        for ticker in tickers:
            try:
                temp_data = ak.stock_zh_a_hist_min_em(
                    symbol=ticker,
                    period=TIME_INTERVAL,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    adjust=ADJUST,
                )
                temp_data["ticker"] = ticker
                df = pd.concat([df, temp_data])
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
    else:
        raise ValueError("Invalid time interval")

    # store data into a csv file
    df.to_csv("raw_data.csv", index=False)
    return df


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
    column_drop = ["change", "turnover", "pct_chg", "amplitude"]
    df["ticker"] = df["ticker"].astype(str)
    df["ticker"] = df["ticker"].apply(lambda x: x.zfill(6))

    df = df.rename(columns=column_mapping)
    df = df.sort_values(by=["date", "tic"]).drop(columns=column_drop)
    # move tic column to the second column
    tic_column = df.pop("tic")
    df.insert(1, "tic", tic_column)

    # remove "-" in date
    # df["date"] = df["date"].apply(lambda x: x.replace("-", ""))

    # add technical indicators
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=True,
        user_defined_feature=True,
    )
    df = fe.preprocess_data(df)

    list_ticker = df["tic"].unique().tolist()
    list_date = list(pd.date_range(df["date"].min(), df["date"].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))
    stock_df = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        df, on=["date", "tic"], how="left"
    )
    stock_df = stock_df[stock_df["date"].isin(df["date"])]
    stock_df = stock_df.sort_values(["date", "tic"])

    # fill the missing values
    stock_df.interpolate(method="linear", inplace=True)
    stock_df.fillna(method="ffill", inplace=True)
    stock_df.fillna(method="bfill", inplace=True)

    # sort the data and reset index
    stock_df.sort_values(["date", "tic"], ignore_index=True)

    # write to csv
    stock_df.to_csv("processed_data.csv", index=False)


if __name__ == "__main__":
    read_from_csv = True
    if read_from_csv:
        df = pd.read_csv("raw_data.csv")
    else:
        df = download_data()
    preprocess_data(df)
