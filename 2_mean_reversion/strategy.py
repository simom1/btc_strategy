"""
均值回归策略
============
逻辑：价格短期偏离均值过大，回归概率高，反向入场

入场条件（做多为例）：
  1. 4h/1h 无强趋势（EMA 排列不整齐，处于震荡区间）
  2. 15m 布林带下轨被触及（价格 <= lower_band）
  3. Z-score <= -2（偏离超过 2 个标准差）
  4. RSI < 35（超卖确认）
  5. 成交量未异常放大（排除放量突破）
  6. 5m 出现反转 K 线确认（收盘价 > 开盘价，即阳线）

止损：入场后价格继续偏离，跌破 lower_band - 0.5x ATR
止盈：价格回归到布林带中轨（20 周期 SMA）
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    """返回 (upper, middle, lower)"""
    middle = sma(series, period)
    std    = series.rolling(window=period).std()
    upper  = middle + std_mult * std
    lower  = middle - std_mult * std
    return upper, middle, lower


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """滚动 Z-score：(price - mean) / std"""
    mean = sma(series, period)
    std  = series.rolling(window=period).std()
    return (series - mean) / std.replace(0, np.nan)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def volume_zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """成交量 Z-score，用于检测异常放量"""
    return zscore(series, period)


def add_indicators(df: pd.DataFrame,
                   bb_period: int = 20,
                   bb_std: float = 2.0,
                   rsi_period: int = 14) -> pd.DataFrame:
    df = df.copy()
    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['atr']   = atr(df)

    df['bb_upper'], df['bb_mid'], df['bb_lower'] = \
        bollinger_bands(df['close'], bb_period, bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']  # 带宽

    df['zscore']     = zscore(df['close'], bb_period)
    df['rsi']        = rsi(df['close'], rsi_period)
    df['vol_zscore'] = volume_zscore(df['volume'], bb_period)
    return df


# ─────────────────────────────────────────
# 趋势强度过滤（4h / 1h 用）
# 均值回归只在震荡市做，强趋势时不操作
# ─────────────────────────────────────────

def is_ranging(df: pd.DataFrame, bb_width_threshold: float = 0.035) -> pd.Series:
    """
    判断是否处于震荡区间：
      - EMA20 和 EMA50 差距 < 0.8%
      - 布林带带宽 < bb_width_threshold
    两个条件都满足才算震荡
    """
    ema_diff  = (df['ema20'] - df['ema50']).abs() / df['ema50']
    no_trend  = ema_diff < 0.008
    narrow_bb = df['bb_width'] < bb_width_threshold
    return no_trend & narrow_bb


# ─────────────────────────────────────────
# 入场信号（15m 用）
# ─────────────────────────────────────────

def entry_signal(df: pd.DataFrame,
                 zscore_threshold: float = 2.0,
                 rsi_long: float = 35,
                 rsi_short: float = 65,
                 vol_zscore_max: float = 2.0) -> pd.Series:
    """
    返回入场信号：
      1  = 做多（超卖回归）
     -1  = 做空（超买回归）
      0  = 无信号

    过滤条件：
      - 成交量 Z-score < vol_zscore_max（排除放量突破）
    """
    vol_ok = df['vol_zscore'] < vol_zscore_max

    long_signal = (
        (df['close'] <= df['bb_lower']) &
        (df['zscore'] <= -zscore_threshold) &
        (df['rsi'] < rsi_long) &
        vol_ok
    )
    short_signal = (
        (df['close'] >= df['bb_upper']) &
        (df['zscore'] >= zscore_threshold) &
        (df['rsi'] > rsi_short) &
        vol_ok
    )

    signal = pd.Series(0, index=df.index)
    signal[long_signal]  = 1
    signal[short_signal] = -1
    return signal


# ─────────────────────────────────────────
# 5m 反转确认
# ─────────────────────────────────────────

def reversal_confirmation(df: pd.DataFrame, direction: int) -> pd.Series:
    """
    做多：5m 出现阳线（收盘 > 开盘）且收盘在前一根低点上方
    做空：5m 出现阴线（收盘 < 开盘）且收盘在前一根高点下方
    """
    if direction == 1:
        return ((df['close'] > df['open']) &
                (df['close'] > df['low'].shift(1))).astype(int)
    else:
        return ((df['close'] < df['open']) &
                (df['close'] < df['high'].shift(1))).astype(int) * -1


# ─────────────────────────────────────────
# 止损 / 止盈
# ─────────────────────────────────────────

def calc_stops(entry_price: float, bb_lower: float, bb_upper: float,
               bb_mid: float, atr_value: float, direction: int,
               sl_atr_mult: float = 0.5):
    """
    止盈：回归到布林带中轨
    止损：布林带外轨再延伸 sl_atr_mult * ATR
    """
    if direction == 1:
        take_profit = bb_mid
        stop_loss   = bb_lower - sl_atr_mult * atr_value
    else:
        take_profit = bb_mid
        stop_loss   = bb_upper + sl_atr_mult * atr_value
    return stop_loss, take_profit
