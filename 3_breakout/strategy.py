"""
亚盘突破策略
============
逻辑：
  亚洲盘（00:00-08:00 UTC）价格整理形成区间
  欧美盘开盘（08:00 UTC 后）突破区间方向入场

入场条件（做多为例）：
  1. 当前时间在 08:00-20:00 UTC（欧美盘活跃时段）
  2. 今日亚盘已形成有效区间（high - low < ATR * range_atr_mult）
  3. 1h K线收盘价突破亚盘高点
  4. 突破时成交量 > 亚盘平均成交量 * vol_mult（放量确认）
  5. 5m 收盘价站稳突破位上方（回踩不破）

止损：亚盘区间中点
止盈：突破幅度的 2x（区间高度 * tp_mult）
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────
# 指标
# ─────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['atr'] = atr(df)
    return df


# ─────────────────────────────────────────
# 亚盘区间计算
# 输入：1h 数据，返回每个交易日的亚盘区间
# ─────────────────────────────────────────

def calc_asian_session(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    计算每日亚盘区间（00:00-08:00 UTC）
    返回 DataFrame，index 为日期，列：
      asian_high, asian_low, asian_mid, asian_vol, asian_range
    """
    df = df_1h.copy()
    # 亚盘：hour 0~7
    asian = df[df.index.hour < 8].copy()
    asian['date'] = asian.index.date

    result = asian.groupby('date').agg(
        asian_high=('high',   'max'),
        asian_low= ('low',    'min'),
        asian_vol= ('volume', 'sum'),
    ).reset_index()
    result['asian_mid']   = (result['asian_high'] + result['asian_low']) / 2
    result['asian_range'] = result['asian_high'] - result['asian_low']
    result['date'] = pd.to_datetime(result['date'], utc=True)
    result = result.set_index('date')
    return result


# ─────────────────────────────────────────
# 区间有效性过滤
# ─────────────────────────────────────────

def is_valid_range(asian_range: float, atr_value: float,
                   range_atr_mult: float = 1.5) -> bool:
    """
    亚盘区间不能太宽（宽 = 已经有方向，不是整理）
    区间 < ATR * range_atr_mult 才算有效整理
    """
    return asian_range < atr_value * range_atr_mult


# ─────────────────────────────────────────
# 突破信号（1h 用）
# ─────────────────────────────────────────

def breakout_signal(df_1h: pd.DataFrame,
                    asian_sessions: pd.DataFrame,
                    vol_mult: float = 1.5) -> pd.Series:
    """
    返回每根 1h K线的突破信号：
      1  = 向上突破亚盘高点
     -1  = 向下突破亚盘低点
      0  = 无信号

    条件：
      - 欧美盘时段（08:00-20:00 UTC）
      - 收盘价突破亚盘高/低点
      - 成交量 > 亚盘平均成交量 * vol_mult
    """
    signal = pd.Series(0, index=df_1h.index)

    for ts, row in df_1h.iterrows():
        # 只在欧美盘时段
        if not (8 <= ts.hour < 20):
            continue

        date = ts.normalize()
        if date not in asian_sessions.index:
            continue

        session = asian_sessions.loc[date]
        asian_high  = session['asian_high']
        asian_low   = session['asian_low']
        asian_vol   = session['asian_vol'] / 8   # 亚盘平均每小时成交量

        vol_ok = row['volume'] > asian_vol * vol_mult

        if row['close'] > asian_high and vol_ok:
            signal[ts] = 1
        elif row['close'] < asian_low and vol_ok:
            signal[ts] = -1

    return signal


# ─────────────────────────────────────────
# 5m 入场确认（突破后回踩不破）
# ─────────────────────────────────────────

def entry_confirmation(df_5m: pd.DataFrame,
                       breakout_level: float,
                       direction: int) -> pd.Series:
    """
    突破后价格回踩，但收盘仍在突破位上方（做多）/ 下方（做空）
    """
    if direction == 1:
        return (df_5m['close'] > breakout_level).astype(int)
    else:
        return (df_5m['close'] < breakout_level).astype(int) * -1


# ─────────────────────────────────────────
# 止损 / 止盈
# ─────────────────────────────────────────

def calc_stops(direction: int,
               asian_high: float,
               asian_low: float,
               asian_mid: float,
               tp_mult: float = 2.0):
    """
    止损：亚盘区间中点（突破失败回到中点就离场）
    止盈：突破幅度 * tp_mult
    """
    asian_range = asian_high - asian_low

    if direction == 1:
        stop_loss   = asian_mid
        take_profit = asian_high + asian_range * tp_mult
    else:
        stop_loss   = asian_mid
        take_profit = asian_low - asian_range * tp_mult

    return stop_loss, take_profit
