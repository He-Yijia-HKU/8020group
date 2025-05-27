from strategyBase import StrategyBase
import pandas as pd
import numpy as np


class SimpleIntradayStrategy(StrategyBase):
	def __init__(self, data_processor,
	             short_window=5,
	             long_window=20,
	             max_position_size=0.1,
	             max_drawdown=0.2,
	             stop_loss_pct=0.02,
	             trailing_stop_pct=0.01,
	             debug=False):
		super().__init__(data_processor, max_position_size, max_drawdown,
		                 stop_loss_pct, trailing_stop_pct)
		self.short_window = short_window
		self.long_window = long_window
		self.debug = debug

	def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
		"""
        修改策略逻辑：每隔20根K线切换一次方向，先买入再卖出，循环往复。
        """
		if self.debug:
			print("\n开始生成交易信号")
			print(f"输入数据列名: {data.columns.tolist()}")
			print(f"输入数据形状: {data.shape}")
			print(f"输入数据示例:\n{data.head()}")
			print(f"短期窗口: {self.short_window}, 长期窗口: {self.long_window}")

		# 确保数据按时间排序
		data = data.sort_values('datetime')

		# 计算移动平均线
		data['short_ma'] = data['close'].rolling(window=self.short_window, min_periods=1).mean()
		data['long_ma'] = data['close'].rolling(window=self.long_window, min_periods=1).mean()

		if self.debug:
			print("\n移动平均线计算结果:")
			print(data[['close', 'short_ma', 'long_ma']].head(10))

		# 生成信号
		signals = pd.DataFrame(index=data.index)
		signals['price'] = data['close']
		signals['direction'] = 0
		signals['quantity'] = 0

		# 每隔20根K线切换一次方向
		for i in range(0, len(data), 20):
			if (i // 20) % 2 == 0:
				# 偶数段买入
				signals.loc[data.index[i], 'direction'] = 1
				signals.loc[data.index[i], 'quantity'] = self.calculate_position_size(data['close'].iloc[i], 1)
			else:
				# 奇数段卖出
				signals.loc[data.index[i], 'direction'] = -1
				signals.loc[data.index[i], 'quantity'] = self.calculate_position_size(data['close'].iloc[i], -1)

		# 移除无效信号
		signals = signals[signals['direction'] != 0]

		if self.debug:
			print(f"\n生成信号数量: {len(signals)}")
			if len(signals) > 0:
				print("信号示例:")
				print(signals.head())
				print("\n买入信号数量:", len(signals[signals['direction'] == 1]))
				print("卖出信号数量:", len(signals[signals['direction'] == -1]))
				print("\n信号详情:")
				for idx, signal in signals.iterrows():
					print(
						f"时间: {idx}, 价格: {signal['price']:.2f}, 方向: {'买入' if signal['direction'] == 1 else '卖出'}, 数量: {signal['quantity']}")

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
