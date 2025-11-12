import pandas as pd
import FinanceDataReader as fdr
import os


def get_stock_data(CONFIGS):
    tickers = CONFIGS.TICKERS
    path = CONFIGS.DATAFRAME_PATH
    for ticker in tickers:
        df = fdr.DataReader(ticker)
        df.to_parquet(os.path.join(path, f"{ticker}.parquet"), engine="pyarrow")


def load_dfs(CONFIGS):
    tickers = CONFIGS.TICKERS
    path = CONFIGS.DATAFRAME_PATH
    return [pd.read_parquet(os.path.join(path, f"{ticker}.parquet"), engine="pyarrow") for ticker in tickers]


if __name__ == "__main__":
    pass
