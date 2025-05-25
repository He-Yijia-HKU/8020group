from dataProcessor import DataProcessor
from simpleStrategy import SimpleIntradayStrategy
from backtestEngine import BacktestEngine
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def main():
	# 创建数据处理器实例
	data_processor = DataProcessor(
		debug=True,  # 开启调试模式
		test_days=10  # 只使用前10天的数据
	)

	# 创建策略实例
	strategy = SimpleIntradayStrategy(
		data_processor=data_processor,
		short_window=5,
		long_window=20,
		max_position_size=0.1,
		max_drawdown=0.2,
		stop_loss_pct=0.02,
		trailing_stop_pct=0.01,
		debug=True  # 开启调试模式
	)

	# 创建回测引擎实例
	engine = BacktestEngine(
		strategy=strategy,
		data_processor=data_processor,
		initial_capital=1000000,
		commission_rate=0.0003,
		slippage=0.0001,
		debug=True  # 开启调试模式
	)

	# 运行回测
	results = engine.run_backtest(is_train=True)
	
	# 打印回测结果摘要
	print("\n回测结果摘要:")
	print(f"总交易次数: {len(results['trades'])}")
	print(f"最终权益: {results['equity_curve'][-1]:.2f}")
	print(f"初始资金: {engine.initial_capital:.2f}")
	print(f"最终现金: {engine.cash:.2f}")
	print(f"最终持仓: {engine.positions}")
	
	print("\n性能指标:")
	for metric, value in results['performance_metrics'].items():
		print(f"{metric}: {value:.4f}")

	# 绘制权益曲线
	plt.figure(figsize=(12, 6))
	plt.plot(results['equity_curve'], label='训练集')
	plt.title('策略权益曲线')
	plt.xlabel('交易天数')
	plt.ylabel('权益')
	plt.legend()
	plt.grid(True)
	plt.show()


if __name__ == "__main__":
	main()
