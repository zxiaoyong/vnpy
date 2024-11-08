from vnpy_ctastrategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)

import math
import numpy as np
from datetime import datetime, time, timedelta

from vnpy.trader.constant import Direction, Offset, Status

# Define the start and end times
trd_start_time = time(9, 45, 0)  # 10:00 AM
trd_end_time = time(14, 45, 0)    # 2:00 PM

def is_between_10_and_14(dt:datetime):

    # Extract the time from the datetime object
    current_time = dt.time()

    # Check if the current_time is between start_time and end_time
    return trd_start_time <= current_time <= trd_end_time

class myBiasStrategy(CtaTemplate):
    """"""
    author = "ZXY"

    # parameters:
    
    # RSI_P: 0-100, 75    开仓时RSI值小于此值
    rsi_p = 75
    # BIAS_P1: 0.01-10, 0.13    开仓时MyBIAS小于0.1
    bias_p1 = 0.13
    # BIAS_P2: 0.1-10, 0.8      开仓时在BIAS_N2个周期内，MyBIAS没有大于BIAS_P2的值
    bias_p2 = 0.8
    # BIAS_N2: 3-100, 10
    bias_n2 = 10
    # OVER_120: 0-1, 1    开仓时是否必须在MA120之上
    #over_120 = 1
    # ROC_P: 0.1-10, 0.35    开仓时最近4bar涨幅/跌幅小于0.35%
    roc_p = 0.35
    # MA60_P: 0-1, 0 开仓是是否必须MA5>MA60

    op_offset_px = 1 # 开仓时回踩均线加价

    # rsi_signal = 20
    # rsi_window = 14
    # fast_window = 5
    # slow_window = 20
    fixed_size = 1

    # rsi_value = 0
    # rsi_long = 0
    # rsi_short = 0
    # fast_ma = 0
    # slow_ma = 0
    # ma_trend = 0
    
    N1 = 5
    N2 = 10
    N3 = 20
    N4 = 30
    N5 = 60
    N6 = 120
    
    ma5 = 0
    ma10 = 0
    ma20 = 0
    ma30 = 0
    ma60 = 0
    ma120 = 0
    
    last_open_trade_time:datetime = None
    last_open_cost:float = None

    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0
    c6 = 0

    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    s5 = 0
    s6 = 0
    
    diff_bias = 0
    rsi1 = 0
    roc_4 = 0
    
    loading_hist_bars:bool = False

    parameters = ["rsi_p", "bias_p1", "bias_p2", "bias_n2", "roc_p", "op_offset_px"]

    variables = ["ma5", "ma10", "ma20", "ma30", "diff_bias", "rsi1", "roc_4",
                 "c1", "c2", "c3", "c4", "c5", "c6"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager(size=150)

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        try:
            self.loading_hist_bars = True
            self.load_bar(10, use_database=True)
        finally:
            self.loading_hist_bars = False

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        # self.bg5.update_bar(bar)
        # self.bg15.update_bar(bar)
        self.cancel_all()

        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return
        
        if self.loading_hist_bars:
            return
        
        ma5_s = am.sma(self.N1, array=True)
        ma10_s = am.sma(self.N2, array=True)
        ma20_s = am.sma(self.N3, array=True)
        ma30_s = am.sma(self.N4, array=True)
        ma60_s = am.sma(self.N5, array=True)
        ma120_s = am.sma(self.N6, array=True)
        
        self.ma5 = ma5_s[-1]
        self.ma10 = ma10_s[-1]
        self.ma20 = ma20_s[-1]
        self.ma30 = ma30_s[-1]
        self.ma60 = ma60_s[-1]
        self.ma120 = ma120_s[-1]
        
        diff_bias_s:np.ndarray = self.calc_bias_diff(ma5_s, ma10_s, ma20_s)
        self.diff_bias = diff_bias_s[-1]
        self.rsi1 = am.rsi(self.N1)
        self.roc_4 = am.roc(4)
        
        ma5_is_up = self.ma_up(ma5_s)
        ma10_is_up = self.ma_up(ma10_s, 2)
        ma20_is_up = self.ma_up(ma20_s, 2)
        ma30_is_up = self.ma_up(ma30_s, 3)
        close_above_ma20 = bar.close_price > self.ma20
        close_above_ma60 = bar.close_price > self.ma60
        ma60_above_ma120 = self.ma60 > self.ma120 * 0.999
        # 中短期均线向上
        self.c1 = int(ma5_is_up and ma10_is_up and ma20_is_up and ma30_is_up and
                      close_above_ma20 and close_above_ma60 and self.ma20 > self.ma120
                      and ma60_above_ma120)

        # diff_bias 在N1周期内至少1个小于bias_p1
        c2_count = self.count_pred(diff_bias_s < self.bias_p1, self.N1)
        self.c2 = int(c2_count > 0)
        
        # rsi(5) 小于 rsi_p
        self.c3 = int(self.rsi1 < self.rsi_p)
        
        # 开仓时在BIAS_N2个周期内，MyBIAS没有大于BIAS_P2的值
        c4_count = self.count_pred(diff_bias_s > self.bias_p2, self.N2)
        self.c4 = int(c4_count < 1)
        
        close_above_ma120 = bar.close_price > self.ma120
        # 价格位于中长期均线之上
        self.c5 = int(close_above_ma120)

        # 开仓时最近4bar涨幅/跌幅小于0.35%
        self.c6 = int(self.roc_4 < self.roc_p)

        # 空头开仓条件 Begin
        ma5_is_down = self.ma_down(ma5_s)
        ma10_is_down = self.ma_down(ma10_s, 2)
        ma20_is_down = self.ma_down(ma20_s, 2)
        ma30_is_down = self.ma_down(ma30_s, 3)
        close_below_ma20 = bar.close_price < self.ma20
        close_below_ma60 = bar.close_price < self.ma60
        ma60_below_ma120 = self.ma60 < self.ma120 * 1.001
        # 中短期均线向向下
        self.s1 = int(ma5_is_down and ma10_is_down and ma20_is_down and ma30_is_down and
                      close_below_ma20 and close_below_ma60 and self.ma20 < self.ma120
                      and ma60_below_ma120)

        # diff_bias 在N1周期内至少1个小于bias_p1
        s2_count = self.count_pred(diff_bias_s < self.bias_p1, self.N1)
        self.s2 = int(s2_count > 0)
        
        # rsi(5) 大于 95-rsi_p
        self.s3 = int(self.rsi1 > (95 - self.rsi_p))
        
        # 开仓时在BIAS_N2个周期内，MyBIAS没有大于BIAS_P2的值
        s4_count = self.count_pred(diff_bias_s > self.bias_p2, self.N2)
        self.s4 = int(s4_count < 1)
        
        close_below_ma120 = bar.close_price < self.ma120
        # 价格位于中长期均线之上
        self.s5 = int(close_below_ma120)

        # 开仓时最近4bar跌幅小于-0.35%
        self.s6 = int(self.roc_4 > -self.roc_p)
        # 空头开仓条件 End
        
        if self.pos == 0:
            # print(f"{bar.datetime} c1:{self.c1} c2:{self.c2} c3:{self.c3} c4:{self.c4} c5:{self.c5} c6:{self.c6}")
            if self.c1 and self.c2 and self.c3 and self.c4 and self.c5 and self.c6:
                if is_between_10_and_14(bar.datetime):
                    op_px = self.get_open_long_price(bar, self.ma10, self.ma20)
                    self.buy(op_px, self.fixed_size)
                    self.write_log(f"[LONG] buy at {op_px}")
                else:
                    self.write_log("不在交易时间10:00-14:00")
            elif self.s1 and self.s2 and self.s3 and self.s4 and self.s5 and self.s6:
                if is_between_10_and_14(bar.datetime):
                    # open short position
                    op_px = self.get_open_short_price(bar, self.ma10, self.ma20)
                    self.short(op_px, self.fixed_size)
                    self.write_log(f"[SHORT] sell at {op_px}")
                else:
                    self.write_log("不在交易时间10:00-14:30")

        elif self.pos > 0:

            # 【开仓初期】开仓5分钟内，回撤一定程度需要尽快平仓
            if self.last_open_trade_time is not None and self.last_open_cost is not None:
                own_pos_duration = bar.datetime - self.last_open_trade_time
                if own_pos_duration > timedelta(minutes=2) and own_pos_duration < timedelta(minutes=5):
                    # 开仓后的第3、4分钟检查，如果下跌超过6个点，尽快止损
                    if self.last_open_cost - bar.close_price > 4.8:
                        self.sell(bar.close_price, abs(self.pos))
                        self.write_log(f"[stop loss] close position at {bar.close_price}")
                        self.put_event()
                        return

            # 【开仓中期】MA20开始渐渐上行，只要不连续2bar低于MA20就保持持仓
            # 【止盈】RSI > 80 以后，一旦不上涨就止盈，止盈价格为前一bar close
            if self.rsi1 > 80 and self.ma_down(self.am.close):
                self.sell(bar.close_price, abs(self.pos))
                self.write_log(f"[stop gain 1] close position at {bar.close_price}")
                self.put_event()
                return
            
            # 【止盈】RSI > 86， bar close时止盈
            if self.rsi1 > 86:
                self.sell(bar.close_price, abs(self.pos))
                self.write_log(f"[stop gain 2] close position at {bar.close_price}")
                self.put_event()
                return
            
            # 连续2bar低于ma20
            sell_cond:bool = self.count_pred(am.close < ma20_s, 2) >= 2
            if sell_cond:
                self.sell(bar.close_price, abs(self.pos))
                self.write_log(f"close position at {bar.close_price}")
        
        elif self.pos < 0:
            # 【开空单中期】MA20开始渐渐下行，只要不连续2bar高于MA20就保持持仓
            # 【止盈】RSI < 80 以后，一旦不上涨就止盈，止盈价格为前一bar close
            if self.rsi1 < 20 and self.ma_up(self.am.close):
                self.cover(bar.close_price, abs(self.pos))
                self.write_log(f"[stop gain 1] close short position at {bar.close_price}")
                self.put_event()
                return
            
            # 【止盈】RSI < 15， bar close时止盈
            if self.rsi1 < 15:
                self.cover(bar.close_price, abs(self.pos))
                self.write_log(f"[stop gain 2] close short position at {bar.close_price}")
                self.put_event()
                return
            
            # 连续2bar高于ma20
            sell_cond:bool = self.count_pred(am.close > ma20_s, 2) >= 2
            if sell_cond:
                self.cover(bar.close_price, abs(self.pos))
                self.write_log(f"close short position at {bar.close_price}")

        self.put_event()
    
    def get_open_long_price(self, bar:BarData, ma10:float, ma20:float):
        """
        确定开仓价，考虑回踩MA10
        """
        # 取MA10和MA20中较大者, 四舍五入（向上取整）后，再加op_offset_px
        op_px = math.ceil( max(ma10, ma20) ) + self.op_offset_px
        # op_px = round( max(ma10, ma20) ) + self.op_offset_px
        op_px = min(bar.close_price, op_px) # 多单开仓价不超过前收价
        return op_px

    def get_open_short_price(self, bar:BarData, ma10:float, ma20:float):
        """
        确定开仓价，考虑回踩MA10
        """
        # 取MA10和MA20中较小者, 四舍五入（向下取整）后，再减op_offset_px
        op_px = math.floor( min(ma10, ma20) ) - self.op_offset_px
        op_px = max(bar.close_price, op_px) # 多单开仓价不小于前收价
        return op_px
    
    def calc_bias_diff(self, ma1:np.ndarray, ma2:np.ndarray, ma3:np.ndarray) -> np.ndarray:
        close = self.am.close
        bias1 = (close - ma1) / ma1 * 100
        bias2 = (close - ma2) / ma2 * 100
        bias3 = (close - ma3) / ma3 * 100
        avg_bias = (bias1 + bias2 + bias3) / 3
        diff_bias = abs(bias1 - avg_bias) + abs(bias2 - avg_bias) + abs(bias3 - avg_bias)
        return diff_bias

    def count_pred(self, pred:np.ndarray, n:int) -> int:
        arr_len = pred.size
        true_count = 0
        if arr_len < n:
            true_count = np.sum(pred)
        else:
            last_n_elems = pred[-n:]
            true_count = np.sum(last_n_elems)
        return true_count
    
    def ma_up(self, ma:np.ndarray, n:int = 1) -> bool:
        for i in range(1, n + 1):
            is_up:bool = ma[-i] > ma[-(i+1)]
            if not is_up:
                return False
        return True
    
    def ma_down(self, ma:np.ndarray, n:int = 1) -> bool:
        for i in range(1, n + 1):
            is_up:bool = ma[-i] < ma[-(i+1)]
            if not is_up:
                return False
        return True

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        if order.status != Status.SUBMITTING and order.status != Status.NOTTRADED:
            msg:str = f'{order.direction} {order.offset} {order.symbol} {order.status}'
            self.write_log(msg)

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        if trade.direction == Direction.LONG:
            if trade.offset == Offset.OPEN:
                self.last_open_trade_time = trade.datetime
                self.last_open_cost = trade.price
            else:
                # reset last trade info
                self.last_open_trade_time = None
                self.last_open_cost = None
        
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass
