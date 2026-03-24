"""
均值回归策略 - 回测引擎
========================
周期分工：
  4h/1h  → 趋势过滤（只在震荡市操作）
  15m    → 入场信号（布林带 + Z-score + RSI）
  5m     → 反转确认入场

手续费：$1.2 / 笔，固定 0.1 手
"""

import pandas as pd
import numpy as np
from strategy import (add_indicators, is_ranging, entry_signal,
                      reversal_confirmation, calc_stops)


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
    zscore_threshold: float = 2.0,
    rsi_long: float = 35,
    rsi_short: float = 65,
    sl_atr_mult: float = 0.5,
    vol_zscore_max: float = 2.0,
) -> tuple[list, float]:

    # 加指标
    df_4h  = add_indicators(df_4h)
    df_1h  = add_indicators(df_1h)
    df_15m = add_indicators(df_15m)
    df_5m  = add_indicators(df_5m)

    # 震荡过滤（4h + 1h 都要是震荡）
    ranging_4h = is_ranging(df_4h)
    ranging_1h = is_ranging(df_1h)

    # 15m 入场信号
    sig_15m = entry_signal(df_15m, zscore_threshold, rsi_long, rsi_short, vol_zscore_max)

    # forward-fill 到 5m，shift(1) 避免超前
    idx          = df_5m.index
    ranging_4h_ff = resample_signal(ranging_4h, idx).shift(1)
    ranging_1h_ff = resample_signal(ranging_1h, idx).shift(1)
    sig_15m_ff    = resample_signal(sig_15m,    idx).shift(1)

    # 预计算 5m 反转确认
    conf_long  = reversal_confirmation(df_5m,  1)
    conf_short = reversal_confirmation(df_5m, -1)

    capital  = initial_capital
    trades   = []
    position = None

    for i in range(50, len(df_5m) - 1):
        row   = df_5m.iloc[i]
        ts    = df_5m.index[i]
        atr_v = row['atr']

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
        # 1. 高周期震荡过滤
        if not ranging_4h_ff.iloc[i] or not ranging_1h_ff.iloc[i]:
            continue

        next_open = df_5m['open'].iloc[i + 1]

        for direction, conf in [(1, conf_long), (-1, conf_short)]:
            if conf.iloc[i] == 0:
                continue
            if sig_15m_ff.iloc[i] != direction:
                continue

            # 取 15m 对应的布林带值（用 ffill 对齐）
            bb_lower = resample_signal(df_15m['bb_lower'], idx).iloc[i]
            bb_upper = resample_signal(df_15m['bb_upper'], idx).iloc[i]
            bb_mid   = resample_signal(df_15m['bb_mid'],   idx).iloc[i]

            sl, tp = calc_stops(next_open, bb_lower, bb_upper,
                                bb_mid, atr_v, direction, sl_atr_mult)

            if abs(next_open - sl) < 1e-8:
                continue

            position = {
                'direction':  direction,
                'entry':      next_open,
                'sl':         sl,
                'tp':         tp,
                'entry_time': df_5m.index[i + 1],
            }
            break

    return trades, capital


def run_backtest(
    df_4h:  pd.DataFrame,
    df_1h:  pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m:  pd.DataFrame,
    initial_capital: float = 10_000,
    zscore_threshold: float = 2.2,
    rsi_long: float = 32,
    rsi_short: float = 68,
    sl_atr_mult: float = 0.5,
    vol_zscore_max: float = 1.8,     # 更严格成交量过滤
) -> dict:
    trades, _ = _run_segment(
        df_4h, df_1h, df_15m, df_5m,
        initial_capital, zscore_threshold,
        rsi_long, rsi_short, sl_atr_mult, vol_zscore_max
    )
    stats = _calc_stats(trades, initial_capital)
    return {'trades': pd.DataFrame(trades), 'stats': stats}


def walk_forward(
    df_4h:  pd.DataFrame,
    df_1h:  pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m:  pd.DataFrame,
    initial_capital: float = 10_000,
    zscore_threshold: float = 2.2,
    rsi_long: float = 32,
    rsi_short: float = 68,
    sl_atr_mult: float = 0.5,
    vol_zscore_max: float = 1.8,
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
            test_5m,
            capital, zscore_threshold,
            rsi_long, rsi_short, sl_atr_mult, vol_zscore_max,
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
