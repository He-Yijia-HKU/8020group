from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, date
from dataProcessor import DataProcessor
from abc import ABC, abstractmethod


class StrategyBase(ABC):
	def __init__(self,
	             data_processor: DataProcessor,
	             # 风险管理参数
	             max_position_size: float = 0.1,
	             max_drawdown: float = 0.2,
	             stop_loss_pct: float = 0.02,
	             trailing_stop_pct: float = 0.01):
		"""
        策略基类
        
        参数:
            data_processor: DataProcessor实例
            max_position_size: 最大持仓比例
            max_drawdown: 最大回撤限制
            stop_loss_pct: 止损百分比
            trailing_stop_pct: 追踪止损百分比
        """
		self.data_processor = data_processor
		self.initial_capital = 1000000  # 默认初始资金

		# 风险管理参数
		self.max_position_size = max_position_size
		self.max_drawdown = max_drawdown
		self.stop_loss_pct = stop_loss_pct
		self.trailing_stop_pct = trailing_stop_pct

		# 策略状态变量
		self.positions = 0
		self.entry_price = 0
		self.highest_price = 0
		self.lowest_price = float('inf')
		self.peak_equity = self.initial_capital

	@abstractmethod
	def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
		"""
        生成交易信号
        
        参数:
            data: 当前交易数据
        返回:
            包含交易信号的DataFrame
        """
		pass

	def risk_management(self, position: int, price: float, equity: float) -> bool:
		"""
        风险管理检查
        
        参数:
            position: 当前持仓
            price: 当前价格
            equity: 当前权益
        返回:
            是否通过风险检查
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

	def _check_position_size(self, position: int, price: float) -> bool:
		"""
        检查持仓规模
        """
		position_value = abs(position * price)
		return position_value <= self.initial_capital * self.max_position_size

	def _check_drawdown(self, current_equity: float) -> bool:
		"""
        检查回撤
        """
		self.peak_equity = max(self.peak_equity, current_equity)
		drawdown = (self.peak_equity - current_equity) / self.peak_equity
		return drawdown <= self.max_drawdown

	def _check_stop_loss(self, current_price: float) -> bool:
		"""
        检查止损
        """
		if self.positions > 0:  # 多头
			return current_price <= self.entry_price * (1 - self.stop_loss_pct)
		elif self.positions < 0:  # 空头
			return current_price >= self.entry_price * (1 + self.stop_loss_pct)
		return False

	def _check_trailing_stop(self, current_price: float) -> bool:
		"""
        检查追踪止损
        """
		if self.positions > 0:  # 多头
			self.highest_price = max(self.highest_price, current_price)
			return current_price <= self.highest_price * (1 - self.trailing_stop_pct)
		elif self.positions < 0:  # 空头
			self.lowest_price = min(self.lowest_price, current_price)
			return current_price >= self.lowest_price * (1 + self.trailing_stop_pct)
		return False

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

	def calculate_risk_metrics(self, returns: pd.Series) -> dict:
		"""
        计算风险指标
        Args:
            returns: 收益率序列
        Returns:
            包含各项风险指标的字典
        """
		# 确保收益率序列有效
		if len(returns) == 0 or returns.isna().all():
			return {
				'var_95': 0.0,
				'expected_shortfall': 0.0,
				'volatility': 0.0,
				'sharpe_ratio': 0.0,
				'sortino_ratio': 0.0
			}

		# 移除无效值
		returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
		
		if len(returns) == 0:
			return {
				'var_95': 0.0,
				'expected_shortfall': 0.0,
				'volatility': 0.0,
				'sharpe_ratio': 0.0,
				'sortino_ratio': 0.0
			}

		# 计算年化收益率和波动率
		annual_return = returns.mean() * 252
		annual_volatility = returns.std() * np.sqrt(252)
		
		# 计算风险指标
		var_95 = np.percentile(returns, 5)
		expected_shortfall = returns[returns <= var_95].mean()
		
		# 计算夏普比率
		risk_free_rate = 0.02  # 假设无风险利率为2%
		excess_return = annual_return - risk_free_rate
		sharpe_ratio = excess_return / annual_volatility if annual_volatility != 0 else 0
		
		# 计算索提诺比率
		downside_returns = returns[returns < 0]
		downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
		sortino_ratio = excess_return / downside_std if downside_std != 0 else 0
		
		return {
			'var_95': var_95,
			'expected_shortfall': expected_shortfall,
			'volatility': annual_volatility,
			'sharpe_ratio': sharpe_ratio,
			'sortino_ratio': sortino_ratio
		}
