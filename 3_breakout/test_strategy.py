"""
亚盘突破策略 - 测试入口
用法：
  python test_strategy.py btc_1h.csv btc_5m.csv
"""

import sys
import pandas as pd
import numpy as np
from backtest import run_backtest, walk_forward

BINANCE_COLS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades',
    'taker_buy_base', 'taker_buy_quote', 'ignore'
]

def load_binance_csv(path: str) -> pd.DataFrame:
    first = pd.read_csv(path, nrows=1, header=None).iloc[0, 0]
    has_header = not str(first).isdigit()
    df = pd.read_csv(path, header=0 if has_header else None,
                     names=None if has_header else BINANCE_COLS)
    df.columns = BINANCE_COLS
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']]
    return df.astype(float)


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        print("加载真实数据...")
        df_1h = load_binance_csv(sys.argv[1])
        df_5m = load_binance_csv(sys.argv[2])
    else:
        print("生成模拟数据...")
        def make_fake(periods, freq, seed=42):
            np.random.seed(seed)
            idx   = pd.date_range('2023-01-01', periods=periods, freq=freq, tz='UTC')
            close = 30000 + np.cumsum(np.random.randn(periods) * 150)
            high  = close + np.abs(np.random.randn(periods) * 100)
            low   = close - np.abs(np.random.randn(periods) * 100)
            vol   = np.abs(np.random.randn(periods) * 1000 + 5000)
            return pd.DataFrame({'open': close, 'high': high, 'low': low,
                                 'close': close, 'volume': vol,
                                 'quote_volume': vol*close, 'trades': 1000}, index=idx)
        df_1h = make_fake(8000,  '1h',   seed=1)
        df_5m = make_fake(96000, '5min', seed=2)

    print("\n── 单次全量回测 ──")
    result = run_backtest(df_1h, df_5m)
    print("\n=== 全量回测结果 ===")
    for k, v in result['stats'].items():
        print(f"  {k:20s}: {v}")

    if not result['trades'].empty:
        print(f"\n前 5 笔交易：")
        print(result['trades'][['entry_time', 'exit_time', 'direction',
                                 'entry', 'exit', 'pnl', 'result']].head())

    print("\n── Walk-Forward 测试 ──")
    wf = walk_forward(df_1h, df_5m, n_splits=5)

    print("\n各 Fold 明细：")
    print(wf['wf_results'].to_string(index=False))

    from report import generate_report
    generate_report(result, wf, output='report.png', strategy_name='亚盘突破')
