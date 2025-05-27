import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from strategyBase import StrategyBase
import warnings
import traceback

warnings.filterwarnings('ignore')


class HARARIMAStrategy(StrategyBase):
	def __init__(self, data_processor,
	             # HAR parameters
	             minute_lags=5,  # 5分钟
	             hour_lags=30,  # 半小时
	             day_part_lags=240,  # 半个交易日
	             # ARIMA parameters
	             p=55,
	             d=1,
	             q=55,
	             prediction_window=60,
	             threshold_pct=0.001,
	             # Hybrid model weights
	             har_weight=0.5,
	             arima_weight=0.5,
	             # Risk management parameters
	             max_position_size=0.1,
	             max_drawdown=0.2,
	             stop_loss_pct=0.02,
	             trailing_stop_pct=0.01,
	             debug=False):
		"""
		HAR+ARIMA混合策略

		参数:
			data_processor: 数据处理器
			p: ARIMA模型AR项阶数
			d: ARIMA模型差分阶数
			q: ARIMA模型MA项阶数
			prediction_window: 预测窗口大小
			threshold_pct: 交易信号触发阈值
			har_weight: HAR模型权重
			arima_weight: ARIMA模型权重
			max_position_size: 最大持仓比例
			max_drawdown: 最大回撤限制
			stop_loss_pct: 止损百分比
			trailing_stop_pct: 追踪止损百分比
			debug: 是否输出调试信息
		"""
		super().__init__(data_processor, max_position_size, max_drawdown,
		                 stop_loss_pct, trailing_stop_pct)

		# 确保权重和为1
		total_weight = har_weight + arima_weight
		self.har_weight = har_weight / total_weight
		self.arima_weight = arima_weight / total_weight

		# HAR模型参数
		self.minute_lags = minute_lags
		self.hour_lags = hour_lags
		self.day_part_lags = day_part_lags

		# ARIMA模型参数
		self.p = p
		self.d = d
		self.q = q
		self.prediction_window = prediction_window

		# 其他参数
		self.threshold_pct = threshold_pct
		self.debug = debug

	def _calculate_har_features(self, series):
		"""
		计算HAR特征

		参数:
			series: 价格序列
		返回:
			har_features: HAR特征DataFrame
		"""
		# 确保数据足够长以计算所有特征
		if len(series) < self.day_part_lags:
			return None

		# 计算分钟级收益率
		returns = series.pct_change().fillna(0)

		# 计算已实现波动率（分钟级平方收益率）
		rv = returns ** 2

		# 创建特征DataFrame
		har_features = pd.DataFrame(index=series.index)

		# 短期特征（过去n分钟的平均已实现波动率）
		for i in range(1, 6):  # 1-5分钟
			har_features[f'minute_rv_{i}'] = rv.rolling(window=i).mean().shift(1)

		# 中期特征（过去n小时的平均已实现波动率）
		for i in range(1, 5):  # 1-4小时 (每小时60分钟)
			har_features[f'hour_rv_{i}'] = rv.rolling(window=i * 60).mean().shift(1)

		# 长期特征（过去半天的平均已实现波动率）
		har_features['half_day_rv'] = rv.rolling(window=self.day_part_lags).mean().shift(1)

		# 移除NaN值
		har_features = har_features.dropna()

		return har_features

	def _fit_har_model(self, features, target):
		"""
		拟合HAR模型

		参数:
			features: HAR特征
			target: 目标变量（价格或收益率）
		返回:
			coef: 系数
			intercept: 截距
		"""
		# 简单线性回归实现
		X = features.values
		y = target.values

		# 添加常数项
		X = np.column_stack((np.ones(X.shape[0]), X))

		# 最小二乘法计算系数
		try:
			# 使用伪逆计算回归系数
			coef = np.linalg.pinv(X.T @ X) @ X.T @ y
			intercept = coef[0]
			coef = coef[1:]
			return coef, intercept
		except:
			if self.debug:
				print("HAR模型拟合失败，可能是因为特征之间存在多重共线性")
			return np.zeros(features.shape[1]), 0

	def _predict_har(self, features, coef, intercept):
		"""
		使用HAR模型进行预测

		参数:
			features: HAR特征
			coef: 系数
			intercept: 截距
		返回:
			预测值
		"""
		return intercept + features.values @ coef

	def _predict_arima(self, train_data):
		"""
		使用ARIMA模型进行预测

		参数:
			train_data: 训练数据
		返回:
			预测值
		"""
		try:
			# 拟合ARIMA模型
			model = ARIMA(train_data, order=(self.p, self.d, self.q))
			model_fit = model.fit()

			# 预测下一个值
			forecast = model_fit.forecast(steps=1)
			predicted_price = float(forecast.iloc[0])
			return predicted_price
		except Exception as e:
			if self.debug:
				print(f"ARIMA预测失败: {str(e)}")
				print(traceback.format_exc())
			return None

	def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
		"""
		使用HAR+ARIMA混合模型生成交易信号

		参数:
			data: 分钟级行情数据
		返回:
			signals: 交易信号DataFrame
		"""
		if self.debug:
			print("\n开始生成HAR+ARIMA混合模型交易信号")
			print(f"输入数据列名: {data.columns.tolist()}")
			print(f"输入数据形状: {data.shape}")
			print(
				f"HAR参数: daily_lags={self.daily_lags}, weekly_lags={self.weekly_lags}, monthly_lags={self.monthly_lags}")
			print(f"ARIMA参数: p={self.p}, d={self.d}, q={self.q}")
			print(f"模型权重: HAR={self.har_weight}, ARIMA={self.arima_weight}")

		# 确保数据按时间排序
		data = data.sort_values('datetime')

		# 创建信号DataFrame
		signals = pd.DataFrame(index=data.index)
		signals['price'] = data['close']
		signals['direction'] = 0
		signals['quantity'] = 0

		# 计算HAR特征
		close_series = data['close']
		har_features = self._calculate_har_features(close_series)

		if har_features is None or len(har_features) < self.prediction_window:
			if self.debug:
				print("无法计算HAR特征，数据不足")
			return signals

		# 对齐索引
		common_idx = data.index.intersection(har_features.index)
		if len(common_idx) == 0:
			if self.debug:
				print("HAR特征与原始数据无法对齐")
			return signals

		data = data.loc[common_idx]
		har_features = har_features.loc[common_idx]

		# 使用滑动窗口进行预测
		for i in range(self.prediction_window, len(data)):
			current_idx = data.index[i]
			current_price = data['close'].iloc[i]

			# 前一个时间点的特征用于当前预测
			if i > 0 and data.index[i - 1] in har_features.index:
				prev_features = har_features.loc[data.index[i - 1]]
			else:
				continue

			# HAR模型预测
			train_idx = data.index[i - self.prediction_window:i]
			train_features = har_features.loc[train_idx]
			train_target = data.loc[train_idx, 'close']

			har_coef, har_intercept = self._fit_har_model(train_features, train_target)
			har_prediction = self._predict_har(prev_features.to_frame().T, har_coef, har_intercept)

			if isinstance(har_prediction, np.ndarray) and len(har_prediction) > 0:
				har_prediction = har_prediction[0]

			# ARIMA模型预测
			train_data = data['close'].iloc[i - self.prediction_window:i]
			arima_prediction = self._predict_arima(train_data)

			# 如果ARIMA预测失败，使用HAR预测
			if arima_prediction is None:
				predicted_price = har_prediction
				weights = [1.0, 0.0]
			else:
				# 混合预测
				predicted_price = (self.har_weight * har_prediction +
				                   self.arima_weight * arima_prediction)
				weights = [self.har_weight, self.arima_weight]

			if self.debug and i % 100 == 0:  # 减少调试输出
				print(f"\n时间点: {current_idx}")
				print(f"当前价格: {current_price:.2f}")
				print(f"HAR预测价格: {har_prediction:.2f}")
				print(f"ARIMA预测价格: {arima_prediction if arima_prediction is not None else 'None'}")
				print(f"混合预测价格: {predicted_price:.2f} (权重: HAR={weights[0]:.2f}, ARIMA={weights[1]:.2f})")
				print(f"预测变化率: {(predicted_price / current_price - 1) * 100:.2f}%")

			# 生成交易信号
			if predicted_price > current_price * (1 + self.threshold_pct):  # 预测价格上涨超过阈值
				signals.loc[current_idx, 'direction'] = 1
				signals.loc[current_idx, 'quantity'] = self.calculate_position_size(current_price, 1)
			elif predicted_price < current_price * (1 - self.threshold_pct):  # 预测价格下跌超过阈值
				signals.loc[current_idx, 'direction'] = -1
				signals.loc[current_idx, 'quantity'] = self.calculate_position_size(current_price, -1)

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
			if self.debug:
				print(f"持仓规模检查失败: 当前持仓={position}, 价格={price}")
			return False

		# 检查回撤
		if not self._check_drawdown(equity):
			if self.debug:
				print(f"回撤检查失败: 当前权益={equity}, 峰值权益={self.peak_equity}")
			return False

		# 检查止损
		if position != 0 and self._check_stop_loss(price):
			if self.debug:
				print(f"止损触发: 当前价格={price}, 入场价格={self.entry_price}")
			return False

		# 检查追踪止损
		if position != 0 and self._check_trailing_stop(price):
			if self.debug:
				print(f"追踪止损触发: 当前价格={price}, " +
				      (f"最高价={self.highest_price}" if position > 0 else f"最低价={self.lowest_price}"))
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
				self.lowest_price = float('inf')
		elif self.positions > 0 and position <= 0:  # 多头平仓或反转
			self.highest_price = 0
		elif self.positions < 0 and position >= 0:  # 空头平仓或反转
			self.lowest_price = float('inf')

		# 更新追踪止损价格
		if position > 0:  # 多头
			self.highest_price = max(self.highest_price, price)
		elif position < 0:  # 空头
			self.lowest_price = min(self.lowest_price, price)

		self.positions = position