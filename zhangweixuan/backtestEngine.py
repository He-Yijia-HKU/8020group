from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, date
from strategyBase import StrategyBase
from dataProcessor import DataProcessor

class BacktestEngine:
    def __init__(self, strategy, data_processor, initial_capital=10000000, commission_rate=0.0004, slippage=0.0002, debug=False):
        """
        初始化回测引擎
        Args:
            strategy: 策略对象
            data_processor: 数据处理器对象
            initial_capital: 初始资金（1000万）
            commission_rate: 手续费率（0.04%，包含交易所和期货公司手续费）
            slippage: 滑点（0.02%，更接近实际市场情况）
            debug: 是否开启调试模式
        """
        self.strategy = strategy
        self.data_processor = data_processor
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.debug = debug
        
        # 回测状态变量
        self.cash = initial_capital
        self.positions = 0
        self.trades = []
        self.equity_curve = [initial_capital]
        self.peak_equity = initial_capital
        self.drawdown = 0
        
    def calculate_slippage(self, price: float, direction: str) -> float:
        """
        计算滑点后的价格
        """
        if direction == 'buy':
            return price * (1 + self.slippage)
        return price * (1 - self.slippage)
    
    def calculate_commission(self, price: float, quantity: int) -> float:
        """
        计算交易手续费
        """
        # 每张合约固定手续费（港币）
        commission_per_contract = 100
        commission_rmb = commission_per_contract * quantity
        return commission_rmb
    
    def _execute_buy(self, price: float, quantity: int) -> None:
        """
        执行买入操作
        """
        total_cost = price * quantity
        commission = self.calculate_commission(price, quantity)
        slippage = self.calculate_slippage(price, 'buy')
        
        if self.debug:
            print(f"尝试买入: 价格={price:.2f}, 数量={quantity}, 总成本={total_cost + commission:.2f}, 当前现金={self.cash:.2f}")
        
        if self.cash >= (total_cost + commission):
            self.positions += quantity
            self.cash -= (total_cost + commission)
            self.trades.append({
                'date': self.current_date,
                'type': 'buy',
                'price': price,
                'slippage_price': slippage,
                'quantity': quantity,
                'commission': commission,
                'total_cost': total_cost + commission,
                'cash_after': self.cash,
                'position_after': self.positions
            })
            if self.debug:
                print(f"买入成功: 价格={price:.2f}, 滑点后={slippage:.2f}, 数量={quantity}, 成本={total_cost + commission:.2f}")
        else:
            if self.debug:
                print(f"买入失败: 资金不足")
    
    def _execute_sell(self, price: float, quantity: int) -> None:
        """
        执行卖出操作
        """
        if self.debug:
            print(f"尝试卖出: 价格={price:.2f}, 数量={quantity}, 当前持仓={self.positions}")
            
        if self.positions >= quantity:
            total_value = price * quantity
            commission = self.calculate_commission(price, quantity)
            slippage = self.calculate_slippage(price, 'sell')
            
            self.positions -= quantity
            self.cash += (total_value - commission)
            self.trades.append({
                'date': self.current_date,
                'type': 'sell',
                'price': price,
                'slippage_price': slippage,
                'quantity': quantity,
                'commission': commission,
                'total_revenue': total_value - commission,
                'cash_after': self.cash,
                'position_after': self.positions
            })
            if self.debug:
                print(f"卖出成功: 价格={price:.2f}, 滑点后={slippage:.2f}, 数量={quantity}, 收入={total_value - commission:.2f}")
        else:
            if self.debug:
                print(f"卖出失败: 持仓不足")
    
    def run_backtest(self, is_train=True):
        """
        运行回测
        Args:
            is_train: 是否使用训练集数据
        """
        if self.debug:
            print(f"\n开始回测 {'训练集' if is_train else '测试集'}")
            print(f"回测日期范围: {self.data_processor.get_all_dates(is_train)[0]} 到 {self.data_processor.get_all_dates(is_train)[-1]}")
            print(f"初始资金: {self.initial_capital:.2f}")

        # 初始化回测状态
        self.cash = self.initial_capital
        self.positions = 0
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.peak_equity = self.initial_capital
        self.drawdown = 0

        # 获取所有交易日
        dates = self.data_processor.get_all_dates(is_train)

        # 按日期遍历
        for date in dates:
            self.current_date = date  # 设置当前日期
            if self.debug:
                print(f"\n处理日期: {date}")

            # 获取当日数据
            daily_data = self.data_processor.get_daily_data(date, is_train)
            if daily_data.empty:
                continue

            if self.debug:
                print(f"数据列: {daily_data.columns.tolist()}")
                print(f"数据形状: {daily_data.shape}")

            # 生成交易信号
            signals = self.strategy.generate_signals(daily_data)
            if self.debug:
                print(f"生成信号数量: {len(signals)}")
                if not signals.empty:
                    print("信号示例:")
                    print(signals.head())

            # 执行交易
            for idx, signal in signals.iterrows():
                price = signal['price']
                quantity = signal['quantity']
                direction = signal['direction']

                if self.debug:
                    print(f"\n处理信号: 价格={price:.2f}, 方向={direction}, 数量={quantity}")

                if direction == 1:  # 买入
                    self._execute_buy(price, quantity)
                elif direction == -1:  # 卖出
                    self._execute_sell(price, quantity)

            # 计算当日权益
            if not daily_data.empty:
                # 使用最后一个价格计算持仓市值
                last_price = daily_data['close'].iloc[-1]
                position_value = self.positions * last_price
                equity = self.cash + position_value
                self.equity_curve.append(equity)

                # 更新最大回撤
                self.peak_equity = max(self.peak_equity, equity)
                current_drawdown = (self.peak_equity - equity) / self.peak_equity
                self.drawdown = max(self.drawdown, current_drawdown)

                if self.debug:
                    print(f"当日结束 - 现金: {self.cash:.2f}, 持仓: {self.positions}, 持仓市值: {position_value:.2f}, 权益: {equity:.2f}")

            # 回测结束时强制平仓
            if self.positions != 0:
                last_price = daily_data['close'].iloc[-1]
                slippage_price = self.calculate_slippage(last_price, 'sell' if self.positions > 0 else 'buy')
                commission = self.calculate_commission(slippage_price, abs(self.positions))
                if self.positions > 0:
                    total_revenue = slippage_price * self.positions - commission
                    self.cash += total_revenue
                    self.trades.append({
                        'date': date,
                        'price': last_price,
                        'quantity': self.positions,
                        'direction': -1,
                        'slippage_price': slippage_price,
                        'commission': commission,
                        'total_cost': 0,
                        'total_revenue': total_revenue
                    })
                else:
                    total_cost = slippage_price * abs(self.positions) + commission
                    self.cash -= total_cost
                    self.trades.append({
                        'date': date,
                        'price': last_price,
                        'quantity': abs(self.positions),
                        'direction': 1,
                        'slippage_price': slippage_price,
                        'commission': commission,
                        'total_cost': total_cost,
                        'total_revenue': 0
                    })
                self.positions = 0
                # 更新最终权益
                self.equity_curve[-1] = self.cash

        if self.debug:
            print("\n回测完成")
            print(f"总交易次数: {len(self.trades)}")
            print(f"最终权益: {self.equity_curve[-1]:.2f}")
        
        # 计算收益率
        equity_array = np.array(self.equity_curve)
        returns = pd.Series(equity_array[1:] / equity_array[:-1] - 1)
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'daily_returns': returns.tolist(),
            'performance_metrics': self.strategy.calculate_risk_metrics(returns)
        }
    
    def plot_results(self):
        """
        绘制回测结果图表
        """
        import matplotlib.pyplot as plt
        
        # 绘制权益曲线
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve)
        plt.title('Equity Curve')
        plt.xlabel('Trading Days')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.show()

# 使用示例
if __name__ == "__main__":
    from strategy_base import StrategyBase
    
    class MyStrategy(StrategyBase):
        def generate_signals(self, data):
            # 实现具体的交易信号生成逻辑
            pass
    
    # 创建策略实例
    strategy = MyStrategy(
        data_processor=DataProcessor(),
        max_position_size=0.1,
        max_drawdown=0.2,
        stop_loss_pct=0.02,
        trailing_stop_pct=0.01
    )
    
    # 创建回测引擎
    engine = BacktestEngine(
        strategy=strategy,
        data_processor=DataProcessor(),
        initial_capital=10000000
    )
    
    # 运行回测
    results = engine.run_backtest(is_train=True)
    
    # 查看结果
    print("回测结果:", results['performance_metrics'])
    
    # 绘制结果
    engine.plot_results()