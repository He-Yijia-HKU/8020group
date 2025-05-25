import pandas as pd
import numpy as np
from datetime import datetime
import os


class DataProcessor:
	def __init__(self, file_path="hi1_20170701_20200609.csv", debug=False, test_days=None):
		"""
        初始化数据处理器
        Args:
            file_path: CSV文件路径
            debug: 是否开启调试模式
            test_days: 回测天数，如果为None则使用全部数据
        """
		self.file_path = file_path
		self.data = None
		self.train_data = None
		self.test_data = None
		self.debug = debug
		self.test_days = test_days
		self._load_data()

	def _load_data(self):
		"""
        加载并预处理数据
        """
		if self.debug:
			print("\n开始加载数据...")
		# 读取CSV文件
		self.data = pd.read_csv(self.file_path)
		if self.debug:
			print(f"原始数据形状: {self.data.shape}")
			print(f"原始数据列名: {self.data.columns.tolist()}")

		# 将日期和时间合并为datetime格式
		self.data['datetime'] = pd.to_datetime(
			self.data['date'].astype(str) + ' ' +
			self.data['time'].astype(str).str.zfill(6),
			format='%Y%m%d %H%M%S'
		)

		# 添加日期列，方便后续按日期分组
		self.data['date'] = self.data['datetime'].dt.date

		# 重命名列以匹配策略需求
		self.data = self.data.rename(columns={
			'hi1_open': 'open',
			'hi1_high': 'high',
			'hi1_low': 'low',
			'hi1_close': 'close',
			'hi1_volume': 'volume'
		})

		# 按日期排序
		self.data = self.data.sort_values('datetime')

		# 如果指定了回测天数，则只保留前N天的数据
		if self.test_days is not None:
			unique_dates = sorted(self.data['date'].unique())
			selected_dates = unique_dates[:self.test_days]
			self.data = self.data[self.data['date'].isin(selected_dates)]
			if self.debug:
				print(f"\n限制回测天数为: {self.test_days}")
				print(f"选择的数据范围: {selected_dates[0]} 到 {selected_dates[-1]}")

		if self.debug:
			print("数据加载完成")
			print(f"处理后的数据形状: {self.data.shape}")
			print(f"处理后的数据列名: {self.data.columns.tolist()}")

	def split_data(self, train_ratio=0.8):
		"""
        将数据分割为训练集和测试集
        Args:
            train_ratio: 训练集比例
        """
		unique_dates = sorted(self.data['date'].unique())
		split_idx = int(len(unique_dates) * train_ratio)
		train_dates = unique_dates[:split_idx]
		test_dates = unique_dates[split_idx:]

		self.train_data = self.data[self.data['date'].isin(train_dates)]
		self.test_data = self.data[self.data['date'].isin(test_dates)]

		return self.train_data, self.test_data

	def get_daily_data(self, date, is_train=True):
		"""
        获取指定日期的数据
        Args:
            date: 日期（datetime.date对象或字符串'YYYY-MM-DD'）
            is_train: 是否从训练集获取数据
        Returns:
            指定日期的数据DataFrame
        """
		if isinstance(date, str):
			date = datetime.strptime(date, '%Y-%m-%d').date()

		if is_train:
			if self.train_data is None:
				self.split_data()
			data = self.train_data[self.train_data['date'] == date]
		else:
			if self.test_data is None:
				self.split_data()
			data = self.test_data[self.test_data['date'] == date]
			
		if self.debug:
			print(f"\n获取{date}的数据")
			print(f"数据形状: {data.shape}")
			print(f"数据列名: {data.columns.tolist()}")
			if not data.empty:
				print("数据示例:")
				print(data.head())
			else:
				print("警告: 该日期没有数据!")
			
		return data

	def get_all_dates(self, is_train=True):
		"""
        获取所有日期列表
        Args:
            is_train: 是否获取训练集的日期
        Returns:
            日期列表
        """
		if is_train:
			if self.train_data is None:
				self.split_data()
			return sorted(self.train_data['date'].unique())
		else:
			if self.test_data is None:
				self.split_data()
			return sorted(self.test_data['date'].unique())


# 使用示例
if __name__ == "__main__":
	# 创建数据处理器实例
	processor = DataProcessor()

	# 分割数据
	train_data, test_data = processor.split_data()

	# 获取特定日期的数据
	sample_date = train_data['date'].iloc[0]
	daily_data = processor.get_daily_data(sample_date, is_train=True)

	print(f"训练集大小: {len(train_data)}")
	print(f"测试集大小: {len(test_data)}")
	print(f"示例日期 {sample_date} 的数据大小: {len(daily_data)}")
