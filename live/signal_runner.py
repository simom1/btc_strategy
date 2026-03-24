"""
实盘信号生成器
==============
每隔 interval_seconds 秒拉一次数据，对所有策略生成信号
只输出信号，不自动下单（需要手动或对接交易接口）

用法：
  python signal_runner.py
  python signal_runner.py --symbol BTCUSDT --interval 60
"""

import sys
import os
import time
import argparse
from datetime import datetime, timezone

# 把各策略目录加入路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, '1_trend_following'))
sys.path.insert(0, os.path.join(ROOT, '2_mean_reversion'))
sys.path.insert(0, os.path.join(ROOT, '3_breakout'))
sys.path.insert(0, os.path.join(ROOT, '4_volume_price'))
sys.path.insert(0, os.path.join(ROOT, '5_market_structure'))

from data_feed import fetch_all_timeframes

# 各策略的信号函数
import importlib

def get_signal_s1(data: dict) -> dict:
    """策略1：多周期趋势跟踪"""
    try:
        import strategy as s1
        df_1d  = s1.add_indicators(data['1d'])
        df_4h  = s1.add_indicators(data['4h'])
        df_1h  = s1.add_indicators(data['1h'])
        df_15m = s1.add_indicators(data['15m'])
        df_5m  = s1.add_indicators(data['5m'])

        dir_1d  = s1.trend_direction(df_1d).iloc[-1]
        dir_4h  = s1.trend_direction(df_4h).iloc[-1]
        sig_1h  = s1.entry_signal(df_1h).iloc[-1]
        sig_15m = s1.entry_signal(df_15m).iloc[-1]
        conf_l  = s1.entry_confirmation(df_5m,  1).iloc[-1]
        conf_s  = s1.entry_confirmation(df_5m, -1).iloc[-1]

        for direction in [1, -1]:
            conf = conf_l if direction == 1 else conf_s
            if conf == 0:
                continue
            aligned = s1.align_timeframes(
                int(dir_1d), int(dir_4h),
                int(sig_1h), int(sig_15m), direction
            )
            if aligned:
                atr_v = df_5m['atr'].iloc[-1]
                price = df_5m['close'].iloc[-1]
                sl, tp = s1.calc_stops(price, atr_v, direction)
                return {'signal': direction, 'price': price, 'sl': sl, 'tp': tp}
    except Exception as e:
        return {'signal': 0, 'error': str(e)}
    return {'signal': 0}


def get_signal_s2(data: dict) -> dict:
    """策略2：均值回归"""
    try:
        import strategy as s2
        df_4h  = s2.add_indicators(data['4h'])
        df_1h  = s2.add_indicators(data['1h'])
        df_15m = s2.add_indicators(data['15m'])
        df_5m  = s2.add_indicators(data['5m'])

        ranging_4h = s2.is_ranging(df_4h).iloc[-1]
        ranging_1h = s2.is_ranging(df_1h).iloc[-1]
        if not ranging_4h or not ranging_1h:
            return {'signal': 0, 'reason': '非震荡市'}

        sig = s2.entry_signal(df_15m).iloc[-1]
        if sig != 0:
            price = df_5m['close'].iloc[-1]
            atr_v = df_5m['atr'].iloc[-1]
            sl, tp = s2.calc_stops(
                price,
                df_15m['bb_lower'].iloc[-1],
                df_15m['bb_upper'].iloc[-1],
                df_15m['bb_mid'].iloc[-1],
                atr_v, int(sig)
            )
            return {'signal': int(sig), 'price': price, 'sl': sl, 'tp': tp}
    except Exception as e:
        return {'signal': 0, 'error': str(e)}
    return {'signal': 0}


def get_signal_s3(data: dict) -> dict:
    """策略3：亚盘突破"""
    try:
        import strategy as s3
        df_1h = s3.add_indicators(data['1h'])
        df_5m = s3.add_indicators(data['5m'])

        asian = s3.calc_asian_session(df_1h)
        sig   = s3.breakout_signal(df_1h, asian)
        last  = sig.iloc[-1]

        if last != 0:
            price = df_5m['close'].iloc[-1]
            today = df_5m.index[-1].normalize()
            if today in asian.index:
                sess = asian.loc[today]
                sl, tp = s3.calc_stops(
                    int(last),
                    sess['asian_high'], sess['asian_low'],
                    sess['asian_mid']
                )
                return {'signal': int(last), 'price': price, 'sl': sl, 'tp': tp}
    except Exception as e:
        return {'signal': 0, 'error': str(e)}
    return {'signal': 0}


def get_signal_s4(data: dict) -> dict:
    """策略4：量价结合"""
    try:
        import strategy as s4
        df_15m = s4.add_indicators(data['15m'])
        df_5m  = s4.add_indicators(data['5m'])

        sig = s4.entry_signal(df_15m).iloc[-1]
        if sig != 0:
            price    = df_5m['close'].iloc[-1]
            swing_lo = df_15m['swing_low'].iloc[-1]
            swing_hi = df_15m['swing_high'].iloc[-1]
            atr_v    = df_15m['atr'].iloc[-1]
            sl, tp   = s4.calc_stops(price, swing_lo, swing_hi,
                                     atr_v, int(sig))
            return {'signal': int(sig), 'price': price, 'sl': sl, 'tp': tp}
    except Exception as e:
        return {'signal': 0, 'error': str(e)}
    return {'signal': 0}


def get_signal_s5(data: dict) -> dict:
    """策略5：市场结构 SMC"""
    try:
        import strategy as s5
        df_4h = s5.add_indicators(data['4h'])
        df_1h = s5.add_indicators(data['1h'])
        df_5m = s5.add_indicators(data['5m'])

        df_4h = s5.find_swing_points(df_4h)
        ms    = s5.market_structure(df_4h)
        direction = int(ms.iloc[-1])
        if direction == 0:
            return {'signal': 0, 'reason': '无明确市场结构'}

        df_1h = s5.find_swing_points(df_1h)
        obs   = s5.find_order_blocks(df_1h)
        if obs.empty:
            return {'signal': 0, 'reason': '无 Order Block'}

        price = df_5m['close'].iloc[-1]
        valid = obs[obs['ob_type'] == direction]
        for _, ob in valid.tail(5).iterrows():
            if s5.price_in_ob(price, ob['ob_high'], ob['ob_low']):
                sl, tp = s5.calc_stops(direction, ob['ob_high'],
                                       ob['ob_low'], price)
                return {'signal': direction, 'price': price, 'sl': sl, 'tp': tp}
    except Exception as e:
        return {'signal': 0, 'error': str(e)}
    return {'signal': 0}


STRATEGIES = {
    '1_trend_following':  (get_signal_s1, '多周期趋势跟踪'),
    '2_mean_reversion':   (get_signal_s2, '均值回归'),
    '3_breakout':         (get_signal_s3, '亚盘突破'),
    '4_volume_price':     (get_signal_s4, '量价结合'),
    '5_market_structure': (get_signal_s5, '市场结构 SMC'),
}

SIGNAL_LABEL = {1: '🟢 做多', -1: '🔴 做空', 0: '⚪ 无信号'}


def run_once(symbol: str = 'BTCUSDT'):
    """拉一次数据，输出所有策略信号"""
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"\n{'='*55}")
    print(f"  BTC 实盘信号  |  {now}")
    print(f"{'='*55}")

    # 切换到各策略目录拉模块
    original_path = sys.path[:]
    data = fetch_all_timeframes(symbol)

    for folder, (fn, name) in STRATEGIES.items():
        # 动态切换策略模块路径
        strategy_dir = os.path.join(ROOT, folder)
        sys.path.insert(0, strategy_dir)

        # 重新加载 strategy 模块（避免缓存）
        if 'strategy' in sys.modules:
            del sys.modules['strategy']

        result = fn(data)
        sig    = result.get('signal', 0)
        label  = SIGNAL_LABEL.get(sig, '⚪')

        print(f"\n  [{name}]  {label}")
        if sig != 0:
            print(f"    入场价: {result.get('price', 0):.2f}")
            print(f"    止损:   {result.get('sl', 0):.2f}")
            print(f"    止盈:   {result.get('tp', 0):.2f}")
        elif 'reason' in result:
            print(f"    原因: {result['reason']}")
        elif 'error' in result:
            print(f"    错误: {result['error']}")

        sys.path = original_path

    print(f"\n{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(description='BTC 实盘信号生成器')
    parser.add_argument('--symbol',   default='BTCUSDT', help='交易对')
    parser.add_argument('--interval', default=60, type=int, help='刷新间隔（秒）')
    parser.add_argument('--once',     action='store_true', help='只运行一次')
    args = parser.parse_args()

    if args.once:
        run_once(args.symbol)
    else:
        print(f"启动实盘信号监控，每 {args.interval} 秒刷新一次")
        print("按 Ctrl+C 停止\n")
        while True:
            try:
                run_once(args.symbol)
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n已停止")
                break
            except Exception as e:
                print(f"错误: {e}，60秒后重试...")
                time.sleep(60)


if __name__ == '__main__':
    main()
