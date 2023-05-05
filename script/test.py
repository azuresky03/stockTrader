import pandas as pd
import os
import sys
from pathlib import Path


def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculates the Moving Average Convergence Divergence (MACD) indicator for a given DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the closing prices for a financial instrument.
        fast_period (int): The number of periods used for the fast exponential moving average (default=12).
        slow_period (int): The number of periods used for the slow exponential moving average (default=26).
        signal_period (int): The number of periods used for the signal line (default=9).

    Returns:
        pandas.DataFrame: A DataFrame containing the MACD indicator, the signal line, and the histogram.
    """
    # Calculate the exponential moving averages
    ema_fast = df['close'].ewm(span=fast_period).mean()
    ema_slow = df['close'].ewm(span=slow_period).mean()

    # Calculate the MACD line
    macd = ema_fast - ema_slow

	#! following value will not be used.
    # Calculate the signal line
    signal_line = macd.ewm(span=signal_period).mean()

    # Calculate the histogram
    histogram = macd - signal_line

    # Create a DataFrame with the MACD, signal line, and histogram
    macd_df = pd.DataFrame({'MACD': macd})

    return macd_df

def calculate_rsi(df, period=14):
    """
    Calculates the Relative Strength Index (RSI) indicator for a given DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the closing prices for a financial instrument.
        period (int): The number of periods used for the RSI calculation (default=14).

    Returns:
        pandas.DataFrame: A DataFrame containing the RSI indicator.
    """
    # Calculate the price change
    delta = df['close'].diff()

    # Get rid of the first row, which will be NaN
    delta = delta[1:]

    # Calculate the gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the average gains and losses
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Calculate the relative strength
    rs = avg_gain / avg_loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    # Create a DataFrame with the RSI
    rsi_df = pd.DataFrame({'RSI': rsi})

    return rsi_df

def calculate_adx(df, period=14):
    """
    Calculates the Average Directional Index (ADX) indicator for a given DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the high, low, and closing prices for a financial instrument.
        period (int): The number of periods used for the ADX calculation (default=14).

    Returns:
        pandas.DataFrame: A DataFrame containing the ADX indicator.
    """
    # Calculate the directional movement
    up_move = df['high'].diff()
    down_move = -df['low'].diff()

    # Get rid of the first row, which will be NaN
    up_move = up_move[1:]
    down_move = down_move[1:]

    # Calculate the true range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Get rid of the first row, which will be NaN
    true_range = true_range[1:]

    # Calculate the positive directional index (DI+)
    up_move[up_move < 0] = 0
    up_di = 100 * (up_move.ewm(alpha=1/period).mean() / true_range.ewm(alpha=1/period).mean())

    # Calculate the negative directional index (DI-)
    down_move[down_move < 0] = 0
    down_di = 100 * (down_move.ewm(alpha=1/period).mean() / true_range.ewm(alpha=1/period).mean())

    # Calculate the directional movement index (DX)
    dx = 100 * abs(up_di - down_di) / (up_di + down_di)

    # Calculate the ADX
    adx = dx.ewm(alpha=1/period).mean()

    # Create a DataFrame with the ADX
    adx_df = pd.DataFrame({'ADX': adx})

    return adx_df


#* example for drop column
# df = df.drop('9-Day EMA',axis=1)

#! some value of MACD could be 0. Will that be problemtic?

# first input will be path of csv file to be transformed
file_path = Path(sys.argv[1])
# get file name
file_name = str(file_path).split('\\')[-1]

# read file to dataframe
df = pd.read_csv(file_path)
# do calculation
macd_df = calculate_macd(df)
rsi_df = calculate_rsi(df)
adx_df = calculate_adx(df)

# add result to original dataframe
df = pd.concat([df, macd_df], axis=1)
df = pd.concat([df, rsi_df], axis=1)
df = pd.concat([df, adx_df], axis=1)

# drop unnecessary columns
df = df.drop(['preclose','adjustflag','tradestatus','pctChg','isST'],axis=1)

# form destination address
target = "..\\" + "result" + "\\" + str(file_name)

#print destination
print(target)

df.to_csv(target, index=False)