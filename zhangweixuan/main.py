from dataProcessor import DataProcessor
from ARIMAStrategy import ARIMAStrategy
from backtestEngine import BacktestEngine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def main():
	# 1. 加载数据
	data_processor = DataProcessor(
		file_path="hi1_20170701_20200609.csv",
		debug=True,
		test_days=7
	)
	df = data_processor.data

	# 2. 创建策略实例
	strategy = ARIMAStrategy(
		data_processor=data_processor,
		p=1,  # AR阶数
		d=1,  # 差分阶数
		q=1,  # MA阶数
		prediction_window=60,  # 预测窗口大小
		threshold_pct=0.001,  # 交易信号触发阈值
		max_position_size=0.1,  # 最大持仓比例
		max_drawdown=0.2,  # 最大回撤限制
		stop_loss_pct=0.02,  # 止损百分比
		trailing_stop_pct=0.01,  # 追踪止损百分比
		debug=False  # 开启调试模式
	)

	# 3. 创建回测引擎
	engine = BacktestEngine(
		strategy=strategy,
		data_processor=data_processor,
		initial_capital=1000000,  # 初始资金100万
		commission_rate=0.0004,  # 手续费率0.04%
		slippage=0.0002,  # 滑点0.02%
		debug=False  # 开启调试模式
	)

	# 4. 运行回测
	results = engine.run_backtest(is_train=True)
	print(results)


if __name__ == "__main__":
	main()
