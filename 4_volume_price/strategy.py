"""
量价结合策略
============
逻辑：成交量异常放大 + 价格突破 = 真实方向性信号
      利用 Binance taker_buy_base 判断主动买卖方向

入场条件（做多为例）：
  1. 15m 成交量 > 20周期均量 * vol_mult（异常放量）
  2. 15m 收盘价突破近 N 根高点（价格突破）
  3. taker_buy_ratio > 0.55（主动买入占比 > 55%，买方主导）
  4. OBV 处于上升趋势（OBV > OBV 的 EMA20）
  5. 价格在 VWAP 上方（做多）/ 下方（做空）
  6. 5m 确认：下一根开盘入场

止损：突破前的摆动低点（近 N 根最低价）
止盈：风险的 2.5 倍
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────
# 指标
# ─────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    滚动 VWAP（每日重置）
    典型价格 * 成交量 的累计 / 累计成交量
    """
    typical = (df['high'] + df['low'] + df['close']) / 3
    df2 = df.copy()
    df2['date'] = df2.index.date
    df2['tp_vol'] = typical * df2['volume']

    vwap_vals = pd.Series(index=df.index, dtype=float)
    for date, group in df2.groupby('date'):
        cum_tp_vol = group['tp_vol'].cumsum()
        cum_vol    = group['volume'].cumsum()
        vwap_vals[group.index] = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap_vals


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume"""
    direction = np.sign(df['close'].diff()).fillna(0)
    return (direction * df['volume']).cumsum()


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """当前成交量 / N周期均量"""
    avg_vol = df['volume'].rolling(window=period).mean()
    return df['volume'] / avg_vol.replace(0, np.nan)


def taker_buy_ratio(df: pd.DataFrame) -> pd.Series:
    """
    主动买入占比 = taker_buy_base / volume
    > 0.5 买方主导，< 0.5 卖方主导
    """
    return df['taker_buy_base'] / df['volume'].replace(0, np.nan)


def swing_high(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """近 period 根 K 线最高价"""
    return df['high'].rolling(window=period).max()


def swing_low(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """近 period 根 K 线最低价"""
    return df['low'].rolling(window=period).min()


def add_indicators(df: pd.DataFrame, vol_period: int = 20,
                   swing_period: int = 10) -> pd.DataFrame:
    df = df.copy()
    df['atr']         = atr(df)
    df['ema20']       = ema(df['close'], 20)
    df['vwap']        = vwap(df)
    df['obv']         = obv(df)
    df['obv_ema']     = ema(df['obv'], 20)
    df['vol_ratio']   = volume_ratio(df, vol_period)
    df['swing_high']  = swing_high(df, swing_period)
    df['swing_low']   = swing_low(df, swing_period)

    if 'taker_buy_base' in df.columns:
        df['tbr'] = taker_buy_ratio(df)
    else:
        df['tbr'] = 0.5   # 没有数据时默认中性

    return df


# ─────────────────────────────────────────
# 入场信号（15m 用）
# ─────────────────────────────────────────

def entry_signal(df: pd.DataFrame,
                 vol_mult: float = 2.0,
                 tbr_long: float = 0.55,
                 tbr_short: float = 0.45) -> pd.Series:
    """
    返回入场信号：
      1  = 做多
     -1  = 做空
      0  = 无信号
    """
    vol_spike = df['vol_ratio'] > vol_mult

    # 价格突破近期高低点（当前收盘 > 前一根的 swing_high）
    breakout_up   = df['close'] > df['swing_high'].shift(1)
    breakout_down = df['close'] < df['swing_low'].shift(1)

    # OBV 趋势
    obv_up   = df['obv'] > df['obv_ema']
    obv_down = df['obv'] < df['obv_ema']

    # VWAP 位置
    above_vwap = df['close'] > df['vwap']
    below_vwap = df['close'] < df['vwap']

    # 主动买卖方向
    buy_dominant  = df['tbr'] > tbr_long
    sell_dominant = df['tbr'] < tbr_short

    long_signal = (
        vol_spike & breakout_up &
        obv_up & above_vwap & buy_dominant
    )
    short_signal = (
        vol_spike & breakout_down &
        obv_down & below_vwap & sell_dominant
    )

    signal = pd.Series(0, index=df.index)
    signal[long_signal]  = 1
    signal[short_signal] = -1
    return signal


# ─────────────────────────────────────────
# 止损 / 止盈
# ─────────────────────────────────────────

def calc_stops(entry_price: float, swing_low_val: float, swing_high_val: float,
               atr_value: float, direction: int,
               sl_atr_mult: float = 1.0, tp_mult: float = 2.5):
    """
    止损：ATR 固定倍数（控制单笔风险）
    止盈：风险距离 * tp_mult
    swing_low/high 作为参考，取两者中较近的
    """
    if direction == 1:
        sl_atr   = entry_price - sl_atr_mult * atr_value
        sl_swing = swing_low_val
        stop_loss   = max(sl_atr, sl_swing)   # 取较近的（较大的）
        risk        = entry_price - stop_loss
        take_profit = entry_price + risk * tp_mult
    else:
        sl_atr   = entry_price + sl_atr_mult * atr_value
        sl_swing = swing_high_val
        stop_loss   = min(sl_atr, sl_swing)   # 取较近的（较小的）
        risk        = stop_loss - entry_price
        take_profit = entry_price - risk * tp_mult

    return stop_loss, take_profit
