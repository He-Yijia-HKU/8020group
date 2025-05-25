# 8020 proj

当前实现了期货交易的回测基本功能

## 项目结构

```
├── backtestEngine.py    # 回测引擎核心实现
├── dataProcessor.py     # 数据处理器
├── strategyBase.py      # 策略基类
├── simpleStrategy.py    # 示例策略实现
└── main.py             # 主程序入口
```

## 核心组件说明

### 1. 数据处理器 (DataProcessor)

`dataProcessor.py` 负责数据的加载和预处理：

- 支持从CSV文件加载期货交易数据
- 提供数据分割功能（训练集/测试集）
- 支持按日期获取数据
- 数据预处理包括：
  - 日期时间格式转换
  - 列名标准化
  - 数据排序
  - 支持限制回测天数

### 2. 策略基类 (StrategyBase)

`strategyBase.py` 定义了策略的基类，提供了以下功能：

- 风险管理功能：
  - 最大持仓规模控制
  - 最大回撤限制
  - 止损机制
  - 追踪止损机制
- 风险指标计算：
  - VaR (Value at Risk)
  - 期望尾部损失 (Expected Shortfall)
  - 波动率
  - 夏普比率
  - 索提诺比率

### 3. 示例策略 (SimpleIntradayStrategy)

`simpleStrategy.py` 实现了一个简单的日内交易策略：

- 基于移动平均线的交易信号生成
- 每20根K线切换一次交易方向
- 支持调试模式，可输出详细的交易信号信息

### 4. 回测引擎 (BacktestEngine)

`backtestEngine.py` 是系统的核心组件，提供完整的回测功能：

- 支持自定义初始资金
- 考虑交易成本：
  - 手续费
  - 滑点
- 提供详细的交易记录
- 支持权益曲线计算
- 包含回测结果可视化功能

### 5. 主程序 (main.py)

`main.py` 提供了系统的使用示例：

- 数据处理器配置
- 策略参数设置
- 回测引擎配置
- 结果展示和可视化