# BTC Strategy Research

基于 Binance BTC/USDT 数据的量化策略研究框架，涵盖五大策略方向，支持多周期分析与 Walk-Forward 验证。

## 项目结构

```
btc_strategy/
├── 1_trend_following/     # 多周期趋势跟踪
├── 2_mean_reversion/      # 均值回归
├── 3_breakout/            # 亚盘突破
├── 4_volume_price/        # 量价结合
├── 5_market_structure/    # 市场结构 SMC
└── data/                  # 本地数据目录（不上传）
```

## 策略概览

### 1. 多周期趋势跟踪 `1_trend_following`
- 日线/4h 判断大方向（EMA 多头/空头排列）
- 1h/15m 找入场信号（MACD 金叉/死叉）
- 5m 精确入场（回踩 EMA20 确认）
- ATR 动态止损，风险回报比 1:2.5

### 2. 均值回归 `2_mean_reversion`
- 布林带 + Z-score 捕捉短期超买超卖
- 15m/5m 周期，成交量过滤假信号
- 适合震荡行情

### 3. 亚盘突破 `3_breakout`
- 亚洲盘（00:00-08:00 UTC）定义整理区间
- 欧美盘开盘突破方向入场
- 1h 定义区间，5m 确认突破

### 4. 量价结合 `4_volume_price`
- 成交量异常放大 + 价格突破组合信号
- VWAP 偏离度过滤
- 5m/15m 周期

### 5. 市场结构 SMC `5_market_structure`
- Higher High / Higher Low 趋势识别
- Order Block + Fair Value Gap 回调入场
- 多周期结构对齐

## 数据格式

使用 Binance K线原始格式（12列）：

```
timestamp, open, high, low, close, volume,
close_time, quote_volume, trades,
taker_buy_base, taker_buy_quote, ignore
```

支持周期：`1m / 5m / 15m / 1h / 4h / 1d`

数据文件放入 `data/` 目录，命名建议：
```
data/btc_1d.csv
data/btc_4h.csv
data/btc_1h.csv
data/btc_15m.csv
data/btc_5m.csv
data/btc_1m.csv
```

## 快速开始

```bash
pip install -r requirements.txt

# 模拟数据验证逻辑
cd 1_trend_following
python test_strategy.py

# 真实数据回测
python test_strategy.py ../data/btc_1d.csv ../data/btc_4h.csv \
    ../data/btc_1h.csv ../data/btc_15m.csv ../data/btc_5m.csv
```

## 回测设计

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 初始资金 | $10,000 | - |
| 手数 | 0.1 手 | 固定 |
| 手续费 | $1.2 / 笔 | 固定单边 |
| 止损 | 1x ATR | 动态 |
| 止盈 | 2.5x ATR | 风险回报 1:2.5 |
| Walk-Forward | 5 折 | train 70% / test 30% |

### 防过拟合机制
- Walk-Forward 滚动验证，OOS 结果不参与参数调整
- 参数数量极简，每个参数有明确经济学含义
- 各 fold 表现稳定性作为策略有效性判断标准

## 依赖

见 `requirements.txt`

## License

MIT
