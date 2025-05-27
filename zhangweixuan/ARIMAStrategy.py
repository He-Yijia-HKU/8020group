import traceback

from strategyBase import StrategyBase
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')


class ARIMAStrategy(StrategyBase):
	def __init__(self, data_processor,
	             p=1,  # AR阶数
	             d=1,  # 差分阶数
	             q=1,  # MA阶数
	             prediction_window=60,  # 预测窗口大小
	             threshold_pct=0.001,  # 交易信号触发阈值（默认1%）
	             max_position_size=0.1,
	             max_drawdown=0.2,
	             stop_loss_pct=0.02,
	             trailing_stop_pct=0.01,
	             debug=False):
		super().__init__(data_processor, max_position_size, max_drawdown,
		                 stop_loss_pct, trailing_stop_pct)
		self.p = p
		self.d = d
		self.q = q
		self.prediction_window = prediction_window
		self.threshold_pct = threshold_pct
		self.debug = debug

	def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
		"""
        使用ARIMA模型生成交易信号
        基于预测价格与当前价格的差异来决定交易方向
        """
		if self.debug:
			print("\n开始生成ARIMA交易信号")
			print(f"输入数据列名: {data.columns.tolist()}")
			print(f"输入数据形状: {data.shape}")
			print(f"ARIMA参数: p={self.p}, d={self.d}, q={self.q}")

		# 确保数据按时间排序
		data = data.sort_values('datetime')

		# 创建信号DataFrame
		signals = pd.DataFrame(index=data.index)
		signals['price'] = data['close']
		signals['direction'] = 0
		signals['quantity'] = 0

		# 使用滑动窗口进行预测
		for i in range(self.prediction_window, len(data)):
			# 获取训练数据
			train_data = data['close'].iloc[i - self.prediction_window:i]

			try:
				# 拟合ARIMA模型
				model = ARIMA(train_data, order=(self.p, self.d, self.q))
				model_fit = model.fit()

				# 预测下一个值
				forecast = model_fit.forecast(steps=1)
				predicted_price = float(forecast.iloc[0])  # 修改这里，使用iloc[0]获取第一个预测值
				current_price = data['close'].iloc[i]

				if self.debug:
					print(f"\n时间点: {data.index[i]}")
					print(f"当前价格: {current_price:.2f}")
					print(f"预测价格: {predicted_price:.2f}")
					print(f"预测变化率: {(predicted_price / current_price - 1) * 100:.2f}%")

				# 生成交易信号
				if predicted_price > current_price * (1 + self.threshold_pct):  # 预测价格上涨超过阈值
					signals.loc[data.index[i], 'direction'] = 1
					signals.loc[data.index[i], 'quantity'] = self.calculate_position_size(current_price, 1)
				elif predicted_price < current_price * (1 - self.threshold_pct):  # 预测价格下跌超过阈值
					signals.loc[data.index[i], 'direction'] = -1
					signals.loc[data.index[i], 'quantity'] = self.calculate_position_size(current_price, -1)

			except Exception as e:
				if self.debug:
					print(f"在时间点 {data.index[i]} 预测时发生错误: {str(e)}")
					print("错误详情:")
					print(traceback.format_exc())
				continue

		# 移除无效信号
		signals = signals[signals['direction'] != 0]

		if self.debug:
			print(f"\n生成信号数量: {len(signals)}")
			if len(signals) > 0:
				print("信号示例:")
				print(signals.head())
				print("\n买入信号数量:", len(signals[signals['direction'] == 1]))
				print("卖出信号数量:", len(signals[signals['direction'] == -1]))

		return signals

	def risk_management(self, position: int, price: float, equity: float) -> bool:
		"""
        风险管理检查
        """
		# 检查持仓规模
		if not self._check_position_size(position, price):
			return False

		# 检查回撤
		if not self._check_drawdown(equity):
			return False

		# 检查止损
		if position != 0 and self._check_stop_loss(price):
			return False

		# 检查追踪止损
		if position != 0 and self._check_trailing_stop(price):
			return False

		return True

	def calculate_position_size(self, price: float, signal: float) -> int:
		"""
        计算开仓数量
        基于当前价格和信号强度计算合适的开仓数量
        """
		# 计算最大可开仓数量
		max_position_value = self.initial_capital * self.max_position_size
		max_quantity = int(max_position_value / price)

		# 根据信号强度调整仓位
		position_ratio = abs(signal)  # 信号强度
		quantity = int(max_quantity * position_ratio)

		# 确保最小交易单位为1
		return max(1, quantity)

	def update_position(self, position: int, price: float) -> None:
		"""
        更新持仓状态
        """
		if self.positions == 0 and position != 0:  # 新开仓
			self.entry_price = price
			if position > 0:
				self.highest_price = price
			else:
				self.lowest_price = price
		self.positions = position
