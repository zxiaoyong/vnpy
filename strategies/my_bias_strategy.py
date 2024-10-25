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

import numpy as np


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
    over_120 = 1
    # ROC_P: 0.1-10, 0.35    开仓时最近4bar涨幅/跌幅小于0.35%
    roc_p = 0.35
    # MA60_P: 0-1, 0 开仓是是否必须MA5>MA60

    # rsi_signal = 20
    # rsi_window = 14
    # fast_window = 5
    # slow_window = 20
    # fixed_size = 1

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
    
    diff_bias = 0
    rsi1 = 0
    roc_4 = 0

    parameters = ["rsi_p", "bias_p1", "bias_p2", "bias_n2", "over_120"]

    variables = ["ma5", "ma10", "ma20", "ma30", "diff_bias", "rsi1", "roc_4"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # self.rsi_long = 50 + self.rsi_signal
        # self.rsi_short = 50 - self.rsi_signal

        # self.bg5 = BarGenerator(self.on_bar, 5, self.on_5min_bar)
        # self.am5 = ArrayManager()

        # self.bg15 = BarGenerator(self.on_bar, 15, self.on_15min_bar)
        # self.am15 = ArrayManager()

        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager()

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bar(120, use_database=True)

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
        # self.bg5.update_tick(tick)

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
        ma10_is_up = self.ma_up(ma10_s)
        ma20_is_up = self.ma_up(ma20_s, 2)
        close_above_ma60 = bar.close_price > self.ma60
        # 中短期均线向上
        c1:bool = ma5_is_up and ma10_is_up and ma20_is_up and close_above_ma60

        # diff_bias 在N1周期内至少1个小于bias_p1
        c2_count = self.count_pred(diff_bias_s < self.bias_p1, self.N1)
        c2:bool = c2_count > 0
        
        # rsi(5) 小于 rsi_p
        c3:bool = self.rsi1 < self.rsi_p
        
        # 开仓时在BIAS_N2个周期内，MyBIAS没有大于BIAS_P2的值
        c4_count = self.count_pred(diff_bias_s > self.bias_p2, self.N2)
        c4:bool = c4_count < 1
        
        close_above_ma120 = bar.close_price > self.ma120
        # 价格位于中长期均线之上
        c5:bool = close_above_ma120

        # 开仓时最近4bar涨幅/跌幅小于0.35%
        c6:bool = self.roc_4 < self.roc_p
        
        self.put_event()
        

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

    def on_5min_bar(self, bar: BarData):
        """"""
        self.cancel_all()

        self.am5.update_bar(bar)
        if not self.am5.inited:
            return

        if not self.ma_trend:
            return

        self.rsi_value = self.am5.rsi(self.rsi_window)

        if self.pos == 0:
            if self.ma_trend > 0 and self.rsi_value >= self.rsi_long:
                self.buy(bar.close_price + 5, self.fixed_size)
            elif self.ma_trend < 0 and self.rsi_value <= self.rsi_short:
                self.short(bar.close_price - 5, self.fixed_size)

        elif self.pos > 0:
            if self.ma_trend < 0 or self.rsi_value < 50:
                self.sell(bar.close_price - 5, abs(self.pos))

        elif self.pos < 0:
            if self.ma_trend > 0 or self.rsi_value > 50:
                self.cover(bar.close_price + 5, abs(self.pos))

        self.put_event()

    def on_15min_bar(self, bar: BarData):
        """"""
        self.am15.update_bar(bar)
        if not self.am15.inited:
            return

        self.fast_ma = self.am15.sma(self.fast_window)
        self.slow_ma = self.am15.sma(self.slow_window)

        if self.fast_ma > self.slow_ma:
            self.ma_trend = 1
        else:
            self.ma_trend = -1

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass
