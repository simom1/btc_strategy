"""
策略分析报告生成器
==================
生成 PNG 报告，包含：
  1. 资金曲线（全量 vs OOS）
  2. 盈亏分布直方图
  3. 月度收益热力图
  4. Walk-Forward 各 fold 指标对比
  5. 关键指标汇总表

用法：
  from report import generate_report
  generate_report(bt_result, wf_result, output='report.png')
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

DARK_BG   = '#0d1117'
CARD_BG   = '#161b22'
GREEN     = '#3fb950'
RED       = '#f85149'
BLUE      = '#58a6ff'
YELLOW    = '#d29922'
TEXT      = '#e6edf3'
SUBTEXT   = '#8b949e'
GRID      = '#21262d'


def _set_style(ax, title=''):
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    ax.spines[:].set_color(GRID)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, pad=8, fontweight='bold')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(SUBTEXT)


def generate_report(
    bt_result: dict,
    wf_result: dict,
    output: str = 'report.png',
    strategy_name: str = '多周期趋势跟踪',
):
    trades    = bt_result['trades']
    stats     = bt_result['stats']
    oos_trades = wf_result['oos_trades']
    wf_df     = wf_result['wf_results']
    oos_stats = wf_result['summary']

    fig = plt.figure(figsize=(16, 20), facecolor=DARK_BG)
    fig.suptitle(f'BTC 策略分析报告 — {strategy_name}',
                 color=TEXT, fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3,
                           top=0.95, bottom=0.04, left=0.07, right=0.97)

    # ── 1. 资金曲线 ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _set_style(ax1, '资金曲线')

    if not trades.empty:
        eq = trades.set_index('exit_time')['capital']
        ax1.plot(eq.index, eq.values, color=BLUE, linewidth=1.5, label='全量回测')
        # 回撤填充
        peak = eq.cummax()
        ax1.fill_between(eq.index, eq.values, peak.values,
                         alpha=0.25, color=RED, label='回撤区间')

    if not oos_trades.empty:
        oos_eq = oos_trades.set_index('exit_time')['capital']
        ax2_twin = ax1.twinx()
        ax2_twin.plot(oos_eq.index, oos_eq.values,
                      color=GREEN, linewidth=1.2, linestyle='--', label='OOS')
        ax2_twin.tick_params(colors=SUBTEXT, labelsize=8)
        ax2_twin.spines[:].set_color(GRID)
        ax2_twin.set_ylabel('OOS 资金 ($)', color=SUBTEXT, fontsize=8)

    ax1.set_ylabel('资金 ($)', color=SUBTEXT, fontsize=8)
    ax1.legend(loc='upper left', fontsize=8,
               facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # ── 2. 盈亏分布 ──────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _set_style(ax2, '盈亏分布')
    if not trades.empty:
        wins  = trades[trades['pnl'] > 0]['pnl']
        loses = trades[trades['pnl'] <= 0]['pnl']
        bins  = 30
        ax2.hist(wins.values,  bins=bins, color=GREEN, alpha=0.7, label=f'盈利 {len(wins)}笔')
        ax2.hist(loses.values, bins=bins, color=RED,   alpha=0.7, label=f'亏损 {len(loses)}笔')
        ax2.axvline(0, color=SUBTEXT, linewidth=0.8, linestyle='--')
        ax2.set_xlabel('盈亏 ($)', color=SUBTEXT, fontsize=8)
        ax2.legend(fontsize=8, facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT)

    # ── 3. 月度收益热力图 ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _set_style(ax3, '月度收益 (%)')
    if not trades.empty:
        t = trades.copy()
        t['exit_time'] = pd.to_datetime(t['exit_time'], utc=True)
        t['year']  = t['exit_time'].dt.year
        t['month'] = t['exit_time'].dt.month
        monthly = t.groupby(['year', 'month'])['pnl'].sum().reset_index()

        # 初始资金用第一笔前的 capital 近似
        init_cap = trades['capital'].iloc[0] - trades['pnl'].iloc[0]
        monthly['ret'] = monthly['pnl'] / init_cap * 100

        pivot = monthly.pivot(index='year', columns='month', values='ret').fillna(0)
        pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                         'Jul','Aug','Sep','Oct','Nov','Dec'][:len(pivot.columns)]

        vmax = max(abs(pivot.values.max()), abs(pivot.values.min()), 0.1)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax3.imshow(pivot.values, cmap='RdYlGn', norm=norm, aspect='auto')

        ax3.set_xticks(range(len(pivot.columns)))
        ax3.set_xticklabels(pivot.columns, fontsize=7, color=SUBTEXT)
        ax3.set_yticks(range(len(pivot.index)))
        ax3.set_yticklabels(pivot.index, fontsize=8, color=SUBTEXT)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if val != 0:
                    ax3.text(j, i, f'{val:.1f}%', ha='center', va='center',
                             fontsize=7, color='white' if abs(val) > vmax*0.5 else TEXT)
        plt.colorbar(im, ax=ax3, fraction=0.03, pad=0.04).ax.tick_params(labelcolor=SUBTEXT)

    # ── 4. Walk-Forward fold 对比 ─────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    _set_style(ax4, 'Walk-Forward — 各 Fold 胜率 & PF')
    if not wf_df.empty:
        x = np.arange(len(wf_df))
        w = 0.35
        bars1 = ax4.bar(x - w/2, wf_df['win_rate'], w,
                        color=BLUE, alpha=0.8, label='胜率 (%)')
        bars2 = ax4.bar(x + w/2, wf_df['profit_factor'].clip(upper=10), w,
                        color=YELLOW, alpha=0.8, label='PF (上限10)')
        ax4.axhline(50, color=GREEN, linewidth=0.8, linestyle='--', alpha=0.5)
        ax4.axhline(1,  color=RED,   linewidth=0.8, linestyle='--', alpha=0.5)
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'Fold {i+1}' for i in x], color=SUBTEXT, fontsize=8)
        ax4.legend(fontsize=8, facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT)
        for bar in bars1:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{bar.get_height():.0f}%', ha='center', fontsize=7, color=TEXT)

    # ── 5. Walk-Forward fold 收益 ─────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    _set_style(ax5, 'Walk-Forward — 各 Fold 收益 & 回撤')
    if not wf_df.empty:
        colors = [GREEN if v >= 0 else RED for v in wf_df['total_return']]
        ax5.bar(x - w/2, wf_df['total_return'], w, color=colors, alpha=0.8, label='收益 (%)')
        ax5.bar(x + w/2, wf_df['max_drawdown'], w, color=RED, alpha=0.5, label='最大回撤 (%)')
        ax5.axhline(0, color=SUBTEXT, linewidth=0.8)
        ax5.set_xticks(x)
        ax5.set_xticklabels([f'Fold {i+1}' for i in x], color=SUBTEXT, fontsize=8)
        ax5.legend(fontsize=8, facecolor=CARD_BG, edgecolor=GRID, labelcolor=TEXT)

    # ── 6. 指标汇总表 ─────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_facecolor(CARD_BG)
    ax6.axis('off')
    ax6.set_title('关键指标汇总', color=TEXT, fontsize=9, pad=8, fontweight='bold', loc='left')

    rows = [
        ['指标', '全量回测', 'OOS 汇总', '说明'],
        ['总交易笔数',
         str(stats.get('total_trades', '-')),
         str(oos_stats.get('total_trades', '-')),
         '样本量'],
        ['胜率',
         f"{stats.get('win_rate', 0):.1f}%",
         f"{oos_stats.get('win_rate', 0):.1f}%",
         '> 50% 为正'],
        ['盈利因子 (PF)',
         f"{stats.get('profit_factor', 0):.2f}",
         f"{oos_stats.get('profit_factor', 0):.2f}",
         '> 1.5 可用，> 2 良好'],
        ['平均盈利',
         f"${stats.get('avg_win', 0):.2f}",
         f"${oos_stats.get('avg_win', 0):.2f}",
         ''],
        ['平均亏损',
         f"${stats.get('avg_loss', 0):.2f}",
         f"${oos_stats.get('avg_loss', 0):.2f}",
         ''],
        ['总收益',
         f"{stats.get('total_return', 0):.2f}%",
         f"{oos_stats.get('total_return', 0):.2f}%",
         ''],
        ['最大回撤',
         f"{stats.get('max_drawdown', 0):.2f}%",
         f"{oos_stats.get('max_drawdown', 0):.2f}%",
         '越小越好'],
        ['最终资金',
         f"${stats.get('final_capital', 0):,.2f}",
         f"${oos_stats.get('final_capital', 0):,.2f}",
         '初始 $10,000'],
        ['手续费合计',
         f"${stats.get('total_commission', 0):.2f}",
         f"${oos_stats.get('total_commission', 0):.2f}",
         '$1.2 / 笔'],
    ]

    col_w = [0.18, 0.18, 0.18, 0.46]
    col_x = [0.01, 0.20, 0.39, 0.58]
    row_h = 0.085

    for r_idx, row in enumerate(rows):
        y = 1 - r_idx * row_h
        bg = '#1f2937' if r_idx == 0 else (CARD_BG if r_idx % 2 == 0 else '#1a2030')
        ax6.add_patch(plt.Rectangle((0, y - row_h), 1, row_h,
                                    transform=ax6.transAxes,
                                    facecolor=bg, edgecolor=GRID, linewidth=0.5))
        for c_idx, (cell, cx) in enumerate(zip(row, col_x)):
            color = TEXT if r_idx == 0 else (
                GREEN if ('.' in str(cell) and not str(cell).startswith('-') and c_idx in [1,2])
                else TEXT
            )
            ax6.text(cx + 0.01, y - row_h/2, cell,
                     transform=ax6.transAxes,
                     va='center', fontsize=8.5,
                     color=TEXT if r_idx == 0 else color,
                     fontweight='bold' if r_idx == 0 else 'normal')

    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f'报告已保存：{output}')
