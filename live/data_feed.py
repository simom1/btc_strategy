"""
Binance 实时数据接口
====================
拉取多周期 K 线，返回与回测相同格式的 DataFrame
无需 API Key（公开行情数据）

依赖：pip install python-binance
"""

import pandas as pd
from binance.client import Client

# 公开行情不需要 key，留空即可
client = Client()

INTERVAL_MAP = {
    '1m':  Client.KLINE_INTERVAL_1MINUTE,
    '5m':  Client.KLINE_INTERVAL_5MINUTE,
    '15m': Client.KLINE_INTERVAL_15MINUTE,
    '30m': Client.KLINE_INTERVAL_30MINUTE,
    '1h':  Client.KLINE_INTERVAL_1HOUR,
    '4h':  Client.KLINE_INTERVAL_4HOUR,
    '1d':  Client.KLINE_INTERVAL_1DAY,
}

BINANCE_COLS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades',
    'taker_buy_base', 'taker_buy_quote', 'ignore'
]


def fetch_klines(symbol: str = 'BTCUSDT',
                 interval: str = '1h',
                 limit: int = 500) -> pd.DataFrame:
    """
    拉取最近 limit 根 K 线
    返回标准 OHLCV DataFrame，index 为 UTC DatetimeIndex
    """
    raw = client.get_klines(
        symbol=symbol,
        interval=INTERVAL_MAP[interval],
        limit=limit
    )
    df = pd.DataFrame(raw, columns=BINANCE_COLS)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df = df[['open', 'high', 'low', 'close', 'volume',
             'quote_volume', 'trades', 'taker_buy_base']]
    return df.astype(float)


def fetch_all_timeframes(symbol: str = 'BTCUSDT') -> dict:
    """
    一次性拉取所有策略所需周期
    返回 {'1d': df, '4h': df, '1h': df, '15m': df, '5m': df, '1m': df}
    """
    print(f"拉取 {symbol} 实时数据...")
    data = {}
    limits = {
        '1d':  365,   # 1年日线
        '4h':  500,   # ~83天
        '1h':  500,   # ~20天
        '15m': 500,   # ~5天
        '5m':  500,   # ~1.7天
        '1m':  500,   # ~8小时
    }
    for tf, limit in limits.items():
        data[tf] = fetch_klines(symbol, tf, limit)
        print(f"  {tf}: {len(data[tf])} 根 K线，"
              f"最新: {data[tf].index[-1].strftime('%Y-%m-%d %H:%M')} UTC")
    return data
