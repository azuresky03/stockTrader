import akshare as ak
import pandas as pd
from stockstats import StockDataFrame as Sdf

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
    "boll_ub",
    "boll_lb",
    "rsi_14",
    "rsi_28",
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
        tickers = [ticker[2:] for ticker in tickers]
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


def add_technical_indicator(data):
    df = data.copy()
    df = df.sort_values(by=["tic", "date"])
    stock = Sdf.retype(df.copy())
    unique_ticker = stock.tic.unique()

    for indicator in INDICATORS:
        indicator_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            try:
                temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                temp_indicator = pd.DataFrame(temp_indicator)
                temp_indicator["tic"] = unique_ticker[i]
                temp_indicator["date"] = df[df.tic == unique_ticker[i]][
                    "date"
                ].to_list()
                # indicator_df = indicator_df.append(
                #     temp_indicator, ignore_index=True
                # )
                indicator_df = pd.concat(
                    [indicator_df, temp_indicator], axis=0, ignore_index=True
                )
            except Exception as e:
                print(e)
        df = df.merge(
            indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
        )
    df = df.sort_values(by=["date", "tic"])
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

    # add indicators
    stock_df = add_technical_indicator(df)

    # fill the missing values with zero
    stock_df = stock_df.fillna(0)

    # write to csv
    stock_df.to_csv("processed_data.csv", index=False)


if __name__ == "__main__":
    read_from_csv = True
    if read_from_csv:
        df = pd.read_csv("raw_data.csv")
    else:
        df = download_data()
    preprocess_data(df)
