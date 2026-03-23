"""
用随机模拟数据验证策略逻辑是否跑通
数据下载好后替换 make_fake_df() 为真实数据即可
"""

import pandas as pd
import numpy as np
from backtest import run_backtest


# Binance 原始列名
BINANCE_COLS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades',
    'taker_buy_base', 'taker_buy_quote', 'ignore'
]

def load_binance_csv(path: str) -> pd.DataFrame:
    """
    读取 Binance K线 CSV，返回标准 OHLCV DataFrame
    index 为 UTC DatetimeIndex
    """
    # 自动检测是否有表头
    first = pd.read_csv(path, nrows=1, header=None).iloc[0, 0]
    has_header = not str(first).isdigit()
    df = pd.read_csv(path, header=0 if has_header else None,
                     names=None if has_header else BINANCE_COLS)
    df.columns = BINANCE_COLS
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']]
    df = df.astype(float)
    df['trades'] = df['trades'].astype(int)
    return df


def make_fake_df(periods: int, freq: str, seed: int = 42) -> pd.DataFrame:
    """生成带随机游走的模拟 OHLCV 数据（数据未到位时使用）"""
    np.random.seed(seed)
    idx    = pd.date_range('2023-01-01', periods=periods, freq=freq, tz='UTC')
    close  = 30000 + np.cumsum(np.random.randn(periods) * 200)
    high   = close + np.abs(np.random.randn(periods) * 100)
    low    = close - np.abs(np.random.randn(periods) * 100)
    open_  = close + np.random.randn(periods) * 50
    volume = np.abs(np.random.randn(periods) * 1000 + 5000)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low,
                         'close': close, 'volume': volume}, index=idx)


if __name__ == '__main__':
    import sys

    if len(sys.argv) >= 6:
        # 真实数据模式：python test_strategy.py 1d.csv 4h.csv 1h.csv 15m.csv 5m.csv
        print("加载真实数据...")
        df_1d  = load_binance_csv(sys.argv[1])
        df_4h  = load_binance_csv(sys.argv[2])
        df_1h  = load_binance_csv(sys.argv[3])
        df_15m = load_binance_csv(sys.argv[4])
        df_5m  = load_binance_csv(sys.argv[5])
    else:
        # 模拟数据模式
        print("生成模拟数据...")
        df_1d  = make_fake_df(500,   '1D',    seed=1)
        df_4h  = make_fake_df(2000,  '4h',    seed=2)
        df_1h  = make_fake_df(8000,  '1h',    seed=3)
        df_15m = make_fake_df(32000, '15min', seed=4)
        df_5m  = make_fake_df(96000, '5min',  seed=5)

    from backtest import walk_forward

    print("── 单次全量回测 ──")
    result = run_backtest(df_1d, df_4h, df_1h, df_15m, df_5m)
    print("\n=== 全量回测结果 ===")
    for k, v in result['stats'].items():
        print(f"  {k:20s}: {v}")

    if not result['trades'].empty:
        print(f"\n前 5 笔交易：")
        print(result['trades'][['entry_time','exit_time','direction',
                                 'entry','exit','gross_pnl','commission','pnl','result']].head())

    print("\n── Walk-Forward 测试 ──")
    wf = walk_forward(df_1d, df_4h, df_1h, df_15m, df_5m, n_splits=5)
    print("\n各 Fold 明细：")
    print(wf['wf_results'].to_string(index=False))

    from report import generate_report
    generate_report(result, wf, output='report.png')
