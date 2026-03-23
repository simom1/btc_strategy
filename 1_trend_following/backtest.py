"""
多周期趋势跟踪 - 回测引擎
===========================
数据格式要求（所有周期统一）：
  DataFrame，index 为 UTC DatetimeIndex，列：open / high / low / close / volume

使用方式：
  from backtest import run_backtest, walk_forward
  # 单次回测
  result = run_backtest(df_1d, df_4h, df_1h, df_15m, df_5m)
  # Walk-Forward
  wf = walk_forward(df_1d, df_4h, df_1h, df_15m, df_5m)
  print(wf['summary'])

手续费：每笔开仓 + 平仓各收一次，固定 $1.2，合计每笔交易 $2.4
手数：固定 0.1 手（1 手 = 1 BTC），size 固定 0.1
"""

import pandas as pd
import numpy as np
from strategy import add_indicators, trend_direction, entry_signal, \
                     entry_confirmation, align_timeframes, calc_stops


COMMISSION   = 1.2        # 单边手续费 $1.2
LOT_SIZE     = 0.1        # 固定 0.1 手


# ─────────────────────────────────────────
# 工具：将高周期信号对齐到低周期时间轴
# ─────────────────────────────────────────

def resample_signal(signal: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """把高周期信号 forward-fill 到目标时间轴"""
    return signal.reindex(target_index, method='ffill')


# ─────────────────────────────────────────
# 核心回测（单段数据）
# ─────────────────────────────────────────

def _run_segment(
    df_1d:  pd.DataFrame,
    df_4h:  pd.DataFrame,
    df_1h:  pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m:  pd.DataFrame,
    initial_capital: float,
    sl_mult: float,
    tp_mult: float,
) -> tuple[list, float]:
    """
    在给定数据段上跑回测，返回 (trades_list, final_capital)
    手续费：开仓 $1.2 + 平仓 $1.2 = 每笔 $2.4
    仓位：固定 LOT_SIZE (0.1 BTC)
    """
    # 加指标
    df_1d  = add_indicators(df_1d)
    df_4h  = add_indicators(df_4h)
    df_1h  = add_indicators(df_1h)
    df_15m = add_indicators(df_15m)
    df_5m  = add_indicators(df_5m)

    # 各周期信号
    dir_1d  = trend_direction(df_1d)
    dir_4h  = trend_direction(df_4h)
    sig_1h  = entry_signal(df_1h)
    sig_15m = entry_signal(df_15m)

    # forward-fill 到 5m 时间轴
    idx        = df_5m.index
    dir_1d_ff  = resample_signal(dir_1d,  idx)
    dir_4h_ff  = resample_signal(dir_4h,  idx)
    sig_1h_ff  = resample_signal(sig_1h,  idx)
    sig_15m_ff = resample_signal(sig_15m, idx)

    # 预计算 5m 入场确认（避免循环内重复计算）
    conf_long  = entry_confirmation(df_5m,  1)
    conf_short = entry_confirmation(df_5m, -1)

    capital  = initial_capital
    trades   = []
    position = None

    for i in range(200, len(df_5m)):
        row   = df_5m.iloc[i]
        ts    = df_5m.index[i]
        price = row['close']
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
                    'entry_time':  position['entry_time'],
                    'exit_time':   ts,
                    'direction':   d,
                    'entry':       position['entry'],
                    'exit':        exit_price,
                    'gross_pnl':   round(gross_pnl, 4),
                    'commission':  COMMISSION,
                    'pnl':         round(net_pnl, 4),
                    'result':      'tp' if hit_tp else 'sl',
                    'capital':     round(capital, 4),
                })
                position = None
            continue

        # ── 空仓：检查多周期对齐 ──
        for direction, conf in [(1, conf_long), (-1, conf_short)]:
            if conf.iloc[i] == 0:
                continue

            aligned = align_timeframes(
                daily_dir  = int(dir_1d_ff.iloc[i]),
                h4_dir     = int(dir_4h_ff.iloc[i]),
                h1_signal  = int(sig_1h_ff.iloc[i]),
                m15_signal = int(sig_15m_ff.iloc[i]),
                direction  = direction,
            )
            if not aligned:
                continue

            sl, tp = calc_stops(price, atr_v, direction, sl_mult, tp_mult)
            if abs(price - sl) == 0:
                continue

            # 开仓，手续费平仓时统一扣
            position  = {
                'direction':  direction,
                'entry':      price,
                'sl':         sl,
                'tp':         tp,
                'entry_time': ts,
            }
            break

    return trades, capital


# ─────────────────────────────────────────
# 公开接口：单次全量回测
# ─────────────────────────────────────────

def run_backtest(
    df_1d:  pd.DataFrame,
    df_4h:  pd.DataFrame,
    df_1h:  pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m:  pd.DataFrame,
    initial_capital: float = 10_000,
    sl_mult: float = 1.0,
    tp_mult: float = 2.5,
) -> dict:
    trades, _ = _run_segment(
        df_1d, df_4h, df_1h, df_15m, df_5m,
        initial_capital, sl_mult, tp_mult
    )
    stats = _calc_stats(trades, initial_capital)
    return {'trades': pd.DataFrame(trades), 'stats': stats}


# ─────────────────────────────────────────
# Walk-Forward 测试
# ─────────────────────────────────────────

def walk_forward(
    df_1d:  pd.DataFrame,
    df_4h:  pd.DataFrame,
    df_1h:  pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m:  pd.DataFrame,
    initial_capital: float = 10_000,
    sl_mult: float = 1.0,
    tp_mult: float = 2.5,
    n_splits: int = 5,          # 切几段
    train_ratio: float = 0.7,   # 每段 train 占比
) -> dict:
    """
    滚动 Walk-Forward：
      - 把 df_5m 时间轴切成 n_splits 段
      - 每段前 train_ratio 用于"训练"（此处固定参数，可扩展为参数优化）
      - 后 (1-train_ratio) 为 OOS test，统计 test 段结果
      - 最终汇总所有 test 段的交易，评估策略真实表现

    返回：
      wf_results  - 每个 fold 的详细结果
      oos_trades  - 所有 OOS 交易合并
      summary     - 汇总统计
    """
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

        # 5m 切片
        train_5m = df_5m.iloc[fold_start:train_end]
        test_5m  = df_5m.iloc[train_end:fold_end]

        if len(test_5m) < 400:   # test 段太短跳过
            continue

        # 其他周期按时间范围切片
        def slice_by_time(df, start, end):
            return df[(df.index >= start) & (df.index <= end)]

        t_start = train_5m.index[0]
        t_end   = train_5m.index[-1]
        o_start = test_5m.index[0]
        o_end   = test_5m.index[-1]

        # ── OOS 测试（用固定参数，capital 接续上一 fold）──
        oos_trades, capital = _run_segment(
            slice_by_time(df_1d,  o_start, o_end),
            slice_by_time(df_4h,  o_start, o_end),
            slice_by_time(df_1h,  o_start, o_end),
            slice_by_time(df_15m, o_start, o_end),
            test_5m,
            capital, sl_mult, tp_mult,
        )

        oos_stats = _calc_stats(oos_trades, capital - sum(t['pnl'] for t in oos_trades))

        wf_results.append({
            'fold':        fold + 1,
            'oos_start':   o_start,
            'oos_end':     o_end,
            'trades':      len(oos_trades),
            'win_rate':    oos_stats.get('win_rate', 0),
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


# ─────────────────────────────────────────
# 统计指标
# ─────────────────────────────────────────

def _calc_stats(trades: list, initial_capital: float) -> dict:
    if not trades:
        return {'total_trades': 0}

    df = pd.DataFrame(trades)
    wins  = df[df['pnl'] > 0]
    loses = df[df['pnl'] <= 0]

    final_cap    = df['capital'].iloc[-1]
    total_return = (final_cap - initial_capital) / initial_capital * 100
    win_rate     = len(wins) / len(df) * 100
    avg_win      = wins['pnl'].mean()  if len(wins)  else 0
    avg_loss     = loses['pnl'].mean() if len(loses) else 0
    pf_denom     = abs(loses['pnl'].sum())
    profit_factor = wins['pnl'].sum() / pf_denom if pf_denom != 0 else np.inf

    equity = df['capital']
    peak   = equity.cummax()
    dd     = (equity - peak) / peak * 100
    max_dd = dd.min()

    return {
        'total_trades':  len(df),
        'win_rate':      round(win_rate, 2),
        'profit_factor': round(profit_factor, 2),
        'avg_win':       round(avg_win, 4),
        'avg_loss':      round(avg_loss, 4),
        'total_return':  round(total_return, 2),
        'max_drawdown':  round(max_dd, 2),
        'final_capital': round(final_cap, 2),
        'total_commission': round(len(df) * COMMISSION, 2),
    }
