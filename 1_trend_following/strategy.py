"""
多周期趋势跟踪策略
===================
周期分工：
  日线/4h  - 趋势方向过滤（只在大趋势方向开仓）
  1h/15m   - 入场信号（MACD + EMA 排列）
  5m       - 精确入场（回踩确认）

开仓条件（做多为例）：
  1. 日线 EMA20 > EMA50 > EMA200（多头排列）
  2. 4h  收盘价 > EMA20，MACD 柱状图 > 0
  3. 1h  MACD 金叉，价格在 EMA20 上方
  4. 15m 出现回踩 EMA20 后反弹的 K 线
  5. 5m  入场确认（收盘价重新站上 EMA20）

止损：入场 K 线低点下方 1x ATR
止盈：2.5x ATR 或 遇到反向信号
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range"""
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    """返回 (macd_line, signal_line, histogram)"""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """给任意周期 df 加上所需指标"""
    df = df.copy()
    df['ema20']  = ema(df['close'], 20)
    df['ema50']  = ema(df['close'], 50)
    df['ema200'] = ema(df['close'], 200)
    df['atr']    = atr(df)
    df['macd'], df['macd_signal'], df['macd_hist'] = macd(df['close'])
    return df


# ─────────────────────────────────────────
# 趋势方向判断（日线 / 4h 用）
# ─────────────────────────────────────────

def trend_direction(df: pd.DataFrame) -> pd.Series:
    """
    返回每根 K 线的趋势方向
      1  = 多头（EMA 多头排列 + 收盘在 EMA20 上方）
     -1  = 空头
      0  = 震荡，不操作
    """
    bull = (df['ema20'] > df['ema50']) & \
           (df['ema50'] > df['ema200']) & \
           (df['close'] > df['ema20'])

    bear = (df['ema20'] < df['ema50']) & \
           (df['ema50'] < df['ema200']) & \
           (df['close'] < df['ema20'])

    direction = pd.Series(0, index=df.index)
    direction[bull] = 1
    direction[bear] = -1
    return direction


# ─────────────────────────────────────────
# 入场信号（1h / 15m 用）
# ─────────────────────────────────────────

def entry_signal(df: pd.DataFrame) -> pd.Series:
    """
    返回入场信号
      1  = 做多信号（MACD 金叉 + 价格在 EMA20 上方）
     -1  = 做空信号
      0  = 无信号
    """
    # MACD 金叉：前一根 hist <= 0，当前 hist > 0
    macd_cross_up   = (df['macd_hist'] > 0) & (df['macd_hist'].shift(1) <= 0)
    macd_cross_down = (df['macd_hist'] < 0) & (df['macd_hist'].shift(1) >= 0)

    long_signal  = macd_cross_up  & (df['close'] > df['ema20'])
    short_signal = macd_cross_down & (df['close'] < df['ema20'])

    signal = pd.Series(0, index=df.index)
    signal[long_signal]  = 1
    signal[short_signal] = -1
    return signal


# ─────────────────────────────────────────
# 精确入场确认（5m 用）
# ─────────────────────────────────────────

def entry_confirmation(df: pd.DataFrame, direction: int) -> pd.Series:
    """
    回踩 EMA20 后重新站上（做多）/ 跌破后重新跌回（做空）
    direction: 1 做多, -1 做空
    """
    if direction == 1:
        # 前一根收盘在 EMA20 下方（回踩），当前收盘重新站上
        pullback = df['close'].shift(1) < df['ema20'].shift(1)
        reclaim  = df['close'] > df['ema20']
        return (pullback & reclaim).astype(int)
    else:
        pullback = df['close'].shift(1) > df['ema20'].shift(1)
        reclaim  = df['close'] < df['ema20']
        return (pullback & reclaim).astype(int) * -1


# ─────────────────────────────────────────
# 多周期对齐检查
# ─────────────────────────────────────────

def align_timeframes(
    daily_dir: int,   # 日线方向
    h4_dir: int,      # 4h 方向
    h1_signal: int,   # 1h 入场信号
    m15_signal: int,  # 15m 入场信号
    direction: int    # 目标方向 1 or -1
) -> bool:
    """
    所有周期方向一致才返回 True
    至少需要：日线 + 4h 方向一致，且 1h 或 15m 有信号
    """
    higher_tf_aligned = (daily_dir == direction) and (h4_dir == direction)
    lower_tf_signal   = (h1_signal == direction) or (m15_signal == direction)
    return higher_tf_aligned and lower_tf_signal


# ─────────────────────────────────────────
# 止损 / 止盈计算
# ─────────────────────────────────────────

def calc_stops(entry_price: float, atr_value: float, direction: int,
               sl_mult: float = 1.0, tp_mult: float = 2.5):
    """
    返回 (stop_loss, take_profit)
    默认风险回报比 1:2.5
    """
    if direction == 1:
        stop_loss   = entry_price - sl_mult * atr_value
        take_profit = entry_price + tp_mult * atr_value
    else:
        stop_loss   = entry_price + sl_mult * atr_value
        take_profit = entry_price - tp_mult * atr_value
    return stop_loss, take_profit
