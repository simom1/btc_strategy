"""
市场结构 SMC 策略
=================
核心概念：
  市场结构（MS）：通过摆动高低点判断趋势方向
    上升结构：Higher High + Higher Low (HH/HL)
    下降结构：Lower High + Lower Low (LH/LL)

  Order Block（OB）：机构建仓的 K 线区域
    做多 OB：下降趋势中最后一根阴线（之后价格大幅上涨）
    做空 OB：上升趋势中最后一根阳线（之后价格大幅下跌）

  Fair Value Gap（FVG）：三根 K 线形成的价格缺口
    看涨 FVG：K1 高点 < K3 低点（中间 K2 快速上涨）
    看跌 FVG：K1 低点 > K3 高点（中间 K2 快速下跌）

入场条件（做多为例）：
  1. 4h 市场结构为上升（最近摆动点是 HH/HL）
  2. 1h 识别最近看涨 Order Block
  3. 15m 价格回调进入 OB 区域（high ~ low 之间）
  4. 5m 出现看涨 FVG 或阳线确认

止损：Order Block 低点下方
止盈：下一个结构高点 或 风险的 2x
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────
# 基础指标
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
# 摆动高低点识别
# ─────────────────────────────────────────

def find_swing_points(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """
    识别摆动高点和低点
    swing_high：左右各 left/right 根 K 线的最高点
    swing_low ：左右各 left/right 根 K 线的最低点
    返回带 swing_high / swing_low 列的 DataFrame
    """
    df = df.copy()
    df['swing_high'] = np.nan
    df['swing_low']  = np.nan

    for i in range(left, len(df) - right):
        window_high = df['high'].iloc[i - left: i + right + 1]
        window_low  = df['low'].iloc[i - left: i + right + 1]

        if df['high'].iloc[i] == window_high.max():
            df.iloc[i, df.columns.get_loc('swing_high')] = df['high'].iloc[i]
        if df['low'].iloc[i] == window_low.min():
            df.iloc[i, df.columns.get_loc('swing_low')] = df['low'].iloc[i]

    return df


# ─────────────────────────────────────────
# 市场结构判断
# ─────────────────────────────────────────

def market_structure(df: pd.DataFrame) -> pd.Series:
    """
    基于摆动点判断市场结构
      1  = 上升结构（HH + HL）
     -1  = 下降结构（LH + LL）
      0  = 不明确

    逻辑：取最近两个摆动高点和两个摆动低点比较
    """
    swing_highs = df['swing_high'].dropna()
    swing_lows  = df['swing_low'].dropna()

    structure = pd.Series(0, index=df.index)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return structure

    # 滚动判断：每个时间点看最近的摆动点
    sh_idx = swing_highs.index.tolist()
    sl_idx = swing_lows.index.tolist()

    for i, ts in enumerate(df.index):
        # 找到 ts 之前的摆动高低点
        prev_sh = [x for x in sh_idx if x < ts]
        prev_sl = [x for x in sl_idx if x < ts]

        if len(prev_sh) < 2 or len(prev_sl) < 2:
            continue

        hh = swing_highs[prev_sh[-1]] > swing_highs[prev_sh[-2]]  # 更高的高点
        hl = swing_lows[prev_sl[-1]]  > swing_lows[prev_sl[-2]]   # 更高的低点
        lh = swing_highs[prev_sh[-1]] < swing_highs[prev_sh[-2]]  # 更低的高点
        ll = swing_lows[prev_sl[-1]]  < swing_lows[prev_sl[-2]]   # 更低的低点

        if hh and hl:
            structure[ts] = 1
        elif lh and ll:
            structure[ts] = -1

    return structure


# ─────────────────────────────────────────
# Order Block 识别
# ─────────────────────────────────────────

def find_order_blocks(df: pd.DataFrame, impulse_mult: float = 1.5) -> pd.DataFrame:
    """
    识别 Order Block：
      看涨 OB：下降段中最后一根阴线，之后出现 impulse_mult * ATR 的上涨
      看跌 OB：上升段中最后一根阳线，之后出现 impulse_mult * ATR 的下跌

    返回 DataFrame，每行代表一个 OB：
      ob_type (1=看涨/-1=看跌), ob_high, ob_low, ob_time
    """
    obs = []

    for i in range(1, len(df) - 3):
        candle = df.iloc[i]
        atr_v  = df['atr'].iloc[i]

        # 看涨 OB：阴线 + 之后 3 根内有大幅上涨
        if candle['close'] < candle['open']:   # 阴线
            future_high = df['high'].iloc[i+1: i+4].max()
            if future_high - candle['low'] > impulse_mult * atr_v:
                obs.append({
                    'ob_type': 1,
                    'ob_high': candle['high'],
                    'ob_low':  candle['low'],
                    'ob_time': df.index[i],
                })

        # 看跌 OB：阳线 + 之后 3 根内有大幅下跌
        elif candle['close'] > candle['open']:  # 阳线
            future_low = df['low'].iloc[i+1: i+4].min()
            if candle['high'] - future_low > impulse_mult * atr_v:
                obs.append({
                    'ob_type': -1,
                    'ob_high': candle['high'],
                    'ob_low':  candle['low'],
                    'ob_time': df.index[i],
                })

    return pd.DataFrame(obs) if obs else pd.DataFrame(
        columns=['ob_type', 'ob_high', 'ob_low', 'ob_time'])


# ─────────────────────────────────────────
# Fair Value Gap 识别
# ─────────────────────────────────────────

def find_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别 FVG（三根 K 线）：
      看涨 FVG：df[i-2].high < df[i].low（K1高点 < K3低点）
      看跌 FVG：df[i-2].low  > df[i].high（K1低点 > K3高点）

    返回 DataFrame：fvg_type, fvg_top, fvg_bottom, fvg_time
    """
    fvgs = []
    for i in range(2, len(df)):
        k1 = df.iloc[i - 2]
        k3 = df.iloc[i]

        # 看涨 FVG
        if k1['high'] < k3['low']:
            fvgs.append({
                'fvg_type':   1,
                'fvg_top':    k3['low'],
                'fvg_bottom': k1['high'],
                'fvg_time':   df.index[i],
            })
        # 看跌 FVG
        elif k1['low'] > k3['high']:
            fvgs.append({
                'fvg_type':   -1,
                'fvg_top':    k1['low'],
                'fvg_bottom': k3['high'],
                'fvg_time':   df.index[i],
            })

    return pd.DataFrame(fvgs) if fvgs else pd.DataFrame(
        columns=['fvg_type', 'fvg_top', 'fvg_bottom', 'fvg_time'])


# ─────────────────────────────────────────
# 入场条件：价格进入 OB 区域
# ─────────────────────────────────────────

def price_in_ob(price: float, ob_high: float, ob_low: float) -> bool:
    return ob_low <= price <= ob_high


# ─────────────────────────────────────────
# 止损 / 止盈
# ─────────────────────────────────────────

def calc_stops(direction: int, ob_high: float, ob_low: float,
               entry_price: float, tp_mult: float = 2.0):
    """
    止损：OB 区域外侧
    止盈：风险距离 * tp_mult
    """
    if direction == 1:
        stop_loss   = ob_low  - (ob_high - ob_low) * 0.1   # OB 低点再下 10%
        risk        = entry_price - stop_loss
        take_profit = entry_price + risk * tp_mult
    else:
        stop_loss   = ob_high + (ob_high - ob_low) * 0.1
        risk        = stop_loss - entry_price
        take_profit = entry_price - risk * tp_mult

    return stop_loss, take_profit
