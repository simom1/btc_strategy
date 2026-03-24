"""
市场结构 SMC 策略 - 回测引擎
==============================
周期分工：
  4h  → 市场结构方向（HH/HL 或 LH/LL）
  1h  → Order Block 识别
  15m → 价格回调进入 OB 区域确认
  5m  → FVG 确认 + 入场

手续费：$1.2 / 笔，固定 0.1 手
"""

import pandas as pd
import numpy as np
from strategy import (add_indicators, find_swing_points, market_structure,
                      find_order_blocks, find_fvg, price_in_ob, calc_stops)


COMMISSION = 1.2
LOT_SIZE   = 0.1


def resample_signal(signal: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    return signal.reindex(target_index, method='ffill')


def _run_segment(
    df_4h:  pd.DataFrame,
    df_1h:  pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m:  pd.DataFrame,
    initial_capital: float,
    swing_left: int = 3,
    swing_right: int = 3,
    impulse_mult: float = 1.5,
    tp_mult: float = 2.0,
    ob_lookback: int = 10,       # 只看最近 N 个 OB
) -> tuple[list, float]:

    # 加指标
    df_4h  = add_indicators(df_4h)
    df_1h  = add_indicators(df_1h)
    df_15m = add_indicators(df_15m)
    df_5m  = add_indicators(df_5m)

    # 4h 摆动点 + 市场结构
    df_4h  = find_swing_points(df_4h,  swing_left, swing_right)
    ms_4h  = market_structure(df_4h)

    # 1h Order Block
    df_1h  = find_swing_points(df_1h,  swing_left, swing_right)
    obs_1h = find_order_blocks(df_1h, impulse_mult)

    # 5m FVG
    fvg_5m = find_fvg(df_5m)

    # 4h 市场结构 forward-fill 到 5m
    idx    = df_5m.index
    ms_ff  = resample_signal(ms_4h, idx).shift(1).fillna(0)

    capital  = initial_capital
    trades   = []
    position = None

    for i in range(20, len(df_5m) - 1):
        row  = df_5m.iloc[i]
        ts   = df_5m.index[i]

        # ── 持仓中：检查止损/止盈 ──
        if position is not None:
            d = position['direction']
            hit_sl = (d ==  1 and row['low']  <= position['sl']) or \
                     (d == -1 and row['high'] >= position['sl'])
            hit_tp = (d ==  1 and row['high'] >= position['tp']) or \
                     (d == -1 and row['low']  <= position['tp'])

            if hit_sl or hit_tp:
                exit_price = position['sl'] if hit_sl else position['tp']
                gross_pnl  = (exit_price - position['entry']) * d * LOT_SIZE
                net_pnl    = gross_pnl - COMMISSION
                capital   += net_pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time':  ts,
                    'direction':  d,
                    'entry':      position['entry'],
                    'exit':       exit_price,
                    'gross_pnl':  round(gross_pnl, 4),
                    'commission': COMMISSION,
                    'pnl':        round(net_pnl, 4),
                    'result':     'tp' if hit_tp else 'sl',
                    'capital':    round(capital, 4),
                })
                position = None
            continue

        # ── 空仓：检查条件 ──
        direction = int(ms_ff.iloc[i])
        if direction == 0:
            continue

        price = row['close']

        # 找最近的同方向 OB（在当前时间之前，且价格未穿越过）
        if obs_1h.empty:
            continue
        valid_obs = obs_1h[
            (obs_1h['ob_type'] == direction) &
            (obs_1h['ob_time'] < ts)
        ].tail(ob_lookback)

        if valid_obs.empty:
            continue

        # 检查价格是否在某个 OB 区域内，且该 OB 未被穿越（失效）
        matched_ob = None
        for _, ob in valid_obs.iterrows():
            # OB 失效：做多 OB 被价格跌破低点；做空 OB 被价格涨破高点
            ob_start_idx = df_5m.index.searchsorted(ob['ob_time'])
            prices_since = df_5m['close'].iloc[ob_start_idx:i]
            if direction == 1 and (prices_since < ob['ob_low']).any():
                continue   # OB 已失效
            if direction == -1 and (prices_since > ob['ob_high']).any():
                continue   # OB 已失效
            if price_in_ob(price, ob['ob_high'], ob['ob_low']):
                matched_ob = ob
                break

        if matched_ob is None:
            continue

        # 5m FVG 确认（最近 5 根内有同方向 FVG）
        if not fvg_5m.empty:
            recent_fvg = fvg_5m[
                (fvg_5m['fvg_type'] == direction) &
                (fvg_5m['fvg_time'] >= df_5m.index[max(0, i-5)]) &
                (fvg_5m['fvg_time'] <= ts)
            ]
            if recent_fvg.empty:
                continue

        next_open = df_5m['open'].iloc[i + 1]
        sl, tp = calc_stops(direction, matched_ob['ob_high'],
                            matched_ob['ob_low'], next_open, tp_mult)

        risk = abs(next_open - sl)
        if risk < 1e-8 or np.isnan(risk):
            continue

        position = {
            'direction':  direction,
            'entry':      next_open,
            'sl':         sl,
            'tp':         tp,
            'entry_time': df_5m.index[i + 1],
        }

    return trades, capital


def run_backtest(
    df_4h:  pd.DataFrame,
    df_1h:  pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m:  pd.DataFrame,
    initial_capital: float = 10_000,
    swing_left: int = 3,
    swing_right: int = 3,
    impulse_mult: float = 2.0,
    tp_mult: float = 2.0,
) -> dict:
    trades, _ = _run_segment(df_4h, df_1h, df_15m, df_5m,
                              initial_capital, swing_left, swing_right,
                              impulse_mult, tp_mult)
    stats = _calc_stats(trades, initial_capital)
    return {'trades': pd.DataFrame(trades), 'stats': stats}


def walk_forward(
    df_4h:  pd.DataFrame,
    df_1h:  pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m:  pd.DataFrame,
    initial_capital: float = 10_000,
    swing_left: int = 3,
    swing_right: int = 3,
    impulse_mult: float = 2.0,
    tp_mult: float = 2.0,
    n_splits: int = 5,
    train_ratio: float = 0.7,
) -> dict:

    total_bars = len(df_5m)
    fold_size  = total_bars // n_splits
    wf_results = []
    all_oos_trades = []
    capital = initial_capital

    print(f"Walk-Forward: {n_splits} folds, train={train_ratio:.0%} / test={1-train_ratio:.0%}")
    print(f"每 fold 约 {fold_size} 根 5m K线 ({fold_size*5/60/24:.1f} 天)\n")

    for fold in range(n_splits):
        fold_start = fold * fold_size
        fold_end   = fold_start + fold_size if fold < n_splits - 1 else total_bars
        train_end  = fold_start + int((fold_end - fold_start) * train_ratio)

        test_5m = df_5m.iloc[train_end:fold_end]
        if len(test_5m) < 400:
            continue

        def slice_by_time(df, start, end):
            return df[(df.index >= start) & (df.index <= end)]

        o_start = test_5m.index[0]
        o_end   = test_5m.index[-1]

        oos_trades, capital = _run_segment(
            slice_by_time(df_4h,  o_start, o_end),
            slice_by_time(df_1h,  o_start, o_end),
            slice_by_time(df_15m, o_start, o_end),
            test_5m, capital,
            swing_left, swing_right, impulse_mult, tp_mult,
        )

        oos_stats = _calc_stats(oos_trades, capital - sum(t['pnl'] for t in oos_trades))
        wf_results.append({
            'fold':          fold + 1,
            'oos_start':     o_start,
            'oos_end':       o_end,
            'trades':        len(oos_trades),
            'win_rate':      oos_stats.get('win_rate', 0),
            'profit_factor': oos_stats.get('profit_factor', 0),
            'total_return':  oos_stats.get('total_return', 0),
            'max_drawdown':  oos_stats.get('max_drawdown', 0),
        })
        all_oos_trades.extend(oos_trades)

        print(f"  Fold {fold+1} OOS [{o_start.date()} ~ {o_end.date()}] "
              f"trades={len(oos_trades)} "
              f"win={oos_stats.get('win_rate',0):.1f}% "
              f"PF={oos_stats.get('profit_factor',0):.2f} "
              f"return={oos_stats.get('total_return',0):.2f}%")

    oos_summary = _calc_stats(all_oos_trades, initial_capital)
    oos_summary['final_capital'] = round(capital, 2)

    print(f"\n=== OOS 汇总 ===")
    for k, v in oos_summary.items():
        print(f"  {k:20s}: {v}")

    return {
        'wf_results':  pd.DataFrame(wf_results),
        'oos_trades':  pd.DataFrame(all_oos_trades),
        'summary':     oos_summary,
    }


def _calc_stats(trades: list, initial_capital: float) -> dict:
    if not trades:
        return {'total_trades': 0}

    df = pd.DataFrame(trades)
    wins  = df[df['pnl'] > 0]
    loses = df[df['pnl'] <= 0]

    final_cap     = df['capital'].iloc[-1]
    total_return  = (final_cap - initial_capital) / initial_capital * 100
    win_rate      = len(wins) / len(df) * 100
    avg_win       = wins['pnl'].mean()  if len(wins)  else 0
    avg_loss      = loses['pnl'].mean() if len(loses) else 0
    pf_denom      = abs(loses['pnl'].sum())
    profit_factor = wins['pnl'].sum() / pf_denom if pf_denom != 0 else np.inf

    equity = df['capital']
    peak   = equity.cummax()
    dd     = (equity - peak) / peak * 100
    max_dd = dd.min()

    return {
        'total_trades':     len(df),
        'win_rate':         round(win_rate, 2),
        'profit_factor':    round(profit_factor, 2),
        'avg_win':          round(avg_win, 4),
        'avg_loss':         round(avg_loss, 4),
        'total_return':     round(total_return, 2),
        'max_drawdown':     round(max_dd, 2),
        'final_capital':    round(final_cap, 2),
        'total_commission': round(len(df) * COMMISSION, 2),
    }
