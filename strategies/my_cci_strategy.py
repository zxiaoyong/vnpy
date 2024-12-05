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
import talib
from vnpy.trader.constant import Direction, Offset, Status

# Define the start and end times
trd_start_time = time(9, 45, 0)  # 9:45 AM
trd_end_time = time(14, 45, 0)    # 2:45 PM
force_close_time = time(14, 55, 0) # 2:55 PM

def is_between_10_and_14(dt:datetime):

    # Extract the time from the datetime object
    current_time = dt.time()

    # Check if the current_time is between start_time and end_time
    return trd_start_time <= current_time <= trd_end_time

def is_market_close(dt:datetime):
    current_time = dt.time()
    return current_time >= force_close_time

class myCciStrategy(CtaTemplate):
    """"""
    author = "ZXY"

    # parameters:
    # DIF_LO_P: 1-500, 30         开仓是SUM_DIFF大于此值
    dif_lower_p = 30
    # DIF_UP_P: 1-500, 80         开仓是SUM_DIFF小于此值
    dif_upper_p = 80
    # CCI_P: 10-500, 75      开仓时CCI值大于此值
    cci_p = 75
    # BIAS_P: 0.01-10, 0.2     开仓时DIFF_BIAS小于此值
    bias_long_p = 0.2
    bias_short_p = 0.14
    # RSI_P: 0-100, 85    开仓时RSI值小于此值
    rsi_p = 85
    
    op_offset_px = 1 # 开仓时回踩均线加价

    fixed_size = 1  # 每次下单量

    N1 = 5
    N2 = 10
    N3 = 20
    N4 = 30
    N5 = 60
    N6 = 120
    N7 = 250
    
    ma5_s:np.ndarray = None
    ma10_s:np.ndarray = None
    ma20_s:np.ndarray = None
    ma30_s:np.ndarray = None
    ma60_s:np.ndarray = None
    ma120_s:np.ndarray = None
    ma250_s:np.ndarray = None
    
    vol_ma20_s:np.ndarray = None
    vol_ma60_s:np.ndarray = None
    
    ma5:float = 0
    ma10:float = 0
    ma20:float = 0
    ma30:float = 0
    ma60:float = 0
    ma120:float = 0
    ma250:float = 0
    
    last_open_trade_time:datetime = None
    last_open_cost:float = None

    dif_sum = 0 # 均线差值之和
    dif_ma = 0 # 均线差值和的Moving Average
    
    cci:float = 0 
    cci_ma:float = 0 # cci 10日平均

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
    
    bias1 = 0
    diff_bias = 0
    rsi1 = 0
    roc_4 = 0
    
    loading_hist_bars:bool = False

    parameters = ["dif_lower_p", "dif_upper_p", "cci_p", "bias_long_p", "bias_short_p", "rsi_p", "op_offset_px"]

    variables = ["ma20", "dif_sum", "cci", "bias1", "diff_bias", "rsi1",
                 "c1", "c2", "c3", "c4", "c5", "c6"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager(size=300)

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
        print(f"direction,datetime,diff_ma,bias1,diff_bias,cci,rsi,macd,vol20,vol10,vol")        

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
        self.cancel_all()

        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return
        
        if self.loading_hist_bars:
            return
        
        self.calc_ma_series()
        
        ### 多头开仓条件 BEGIN
        LC:list[bool] = [False] * 8  # 多头条件列表
        diff_ma = self.calc_diff_sum()
        # SUM_DIFF > 30     -- L6c
        LC[1] = diff_ma > self.dif_lower_p and diff_ma < self.dif_upper_p
        
        cci, cci_ma = self.calc_cci()
        # CCI > 75 AND CCI > CCI_MA     -- L4
        LC[2] = cci > self.cci_p and cci > cci_ma
        
        macd = self.calc_my_macd(self.ma10_s, self.ma30_s, MID_P=10)
        # MACD柱 连续2根数值增长     -- L7
        LC[3] = self.ma_up(macd, 2)
        
        self.rsi1 = self.calc_rsi(self.N1)
        # RSI < 85     -- L5
        LC[4] = self.rsi1 < self.rsi_p
        
        bias1_s, diff_bias_s = self.calc_bias_diff(self.ma5_s, self.ma10_s, self.ma20_s)
        self.bias1 = bias1_s[-1]
        self.diff_bias = diff_bias_s[-1]
        # C1:=ABS(BIAS1)<(BIAS_P-0.1) AND DIFF_BIAS<BIAS_P      -- C1
        LC[5] = abs(self.bias1) < (self.bias_long_p - 0.05) and self.diff_bias < self.bias_long_p

        ma5_is_up = self.ma_up(self.ma5_s)
        ma10_is_up = self.ma_up(self.ma10_s, 2)
        ma20_is_up = self.ma_up(self.ma20_s, 2)
        ma30_is_up = self.ma_up(self.ma30_s, 3)
        close_above_ma20 = bar.close_price > self.ma20
        close_above_ma60 = bar.close_price > self.ma60
        ma20_above_ma60 = self.ma20 > self.ma60
        ma60_above_ma120 = self.ma60 > self.ma120 * 0.999
        
        # 中长期均线多排 MA60 > MA120      -- L1
        LC[6] = ma60_above_ma120
        
        # k线在中长期均线之上 CLOSE > MA20 AND CLOSE > MA60 AND MA20 > MA60     -- L2
        LC[7] = close_above_ma20 and close_above_ma60 and ma20_above_ma60
        
        # 中期均线与长期均线一致       -- L3
        LC[0] = ma20_is_up or ma30_is_up
        
        self.c1 = int(LC[1])
        self.c2 = int(LC[2])
        self.c3 = int(LC[3])
        self.c4 = int(LC[4])
        self.c5 = int(LC[5])
        self.c6 = int(LC[6] and LC[7] and LC[0])
        ### 多头开仓条件 END
        
        ### 空头开仓条件 BEGIN
        SC:list[bool] = [False] * 8  # 多头条件列表
        # SUM_DIFF < -30     -- S6
        SC[1] = diff_ma < -self.dif_lower_p
        # CCI < -75 AND CCI < CCI_MA     -- S4
        SC[2] = cci < -self.cci_p and cci < cci_ma
        # MACD柱 连续2根数值减少     -- S7
        SC[3] = self.ma_down(macd, 2)
        # RSI > 10   S5:=RSI1 > (95-RSI_P)  -- S5
        SC[4] = self.rsi1 > (95 - self.rsi_p)
        # C1:=ABS(BIAS1)< 0.13 AND DIFF_BIAS < 0.14
        SC[5] = abs(self.bias1) < (self.bias_short_p - 0.01) and self.diff_bias < self.bias_short_p

        ma20_is_down = self.ma_down(self.ma20_s, 2)
        ma30_is_down = self.ma_down(self.ma30_s, 3)
        close_below_ma20 = bar.close_price < self.ma20
        close_below_ma60 = bar.close_price < self.ma60
        ma20_below_ma60 = self.ma20 < self.ma60
        ma60_below_ma120 = self.ma60 < self.ma120 * 1.001
        
        # 中长期均线空排 MA60 < MA120      -- S1
        SC[6] = ma60_below_ma120
        
        # k线在中长期均线之下 CLOSE < MA20 AND CLOSE < MA60 AND MA20 < MA60     -- S2
        SC[7] = close_below_ma20 and close_below_ma60 and ma20_below_ma60
        
        # 中期均线与长期均线一致       -- S3
        SC[0] = ma20_is_down or ma30_is_down
        
        self.s1 = int(SC[1])
        self.s2 = int(SC[2])
        self.s3 = int(SC[3])
        self.s4 = int(SC[4])
        self.s5 = int(SC[5])
        self.s6 = int(SC[6] and SC[7] and SC[0])
        ### 空头开仓条件 END
        
        if self.pos == 0:
            # print(f"{bar.datetime} c1:{self.c1} c2:{self.c2} c3:{self.c3} c4:{self.c4} c5:{self.c5} c6:{self.c6}")
            if all(LC):
                if is_between_10_and_14(bar.datetime):
                    op_px = self.get_open_long_price(bar, self.ma10, self.ma20)
                    self.buy(op_px, self.fixed_size)
                    self.write_log(f"[LONG] buy at {op_px}")
                    vol_rate_20, vol_rate_60 = self.calc_vol_rate(bar)
                    print(f"long,{bar.datetime},{diff_ma:.2f},{self.bias1:.2f},{self.diff_bias:.2f},{cci:.2f},{self.rsi1:.2f},{macd[-1]:.2f},{vol_rate_20:.2f},{vol_rate_60:.2f},{bar.volume}")
                else:
                    self.write_log("不在交易时间10:00-14:00")
            elif all(SC):
                if is_between_10_and_14(bar.datetime):
                    # open short position
                    op_px = self.get_open_short_price(bar, self.ma10, self.ma20)
                    self.short(op_px, self.fixed_size)
                    self.write_log(f"[SHORT] sell at {op_px}")
                    vol_rate_20, vol_rate_60 = self.calc_vol_rate(bar)
                    print(f"short,{bar.datetime},{diff_ma:.2f},{self.bias1:.2f},{self.diff_bias:.2f},{cci:.2f},{self.rsi1:.2f},{macd[-1]:.2f},{vol_rate_20:.2f},{vol_rate_60:.2f},{bar.volume}")
                else:
                    self.write_log("不在交易时间10:00-14:30")

        elif self.pos > 0:
            
            if is_market_close(bar.datetime):
                self.sell(bar.close_price, abs(self.pos))
                self.write_log(f"close position before marekt close")
                self.put_event()
                return

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

            # # 【开仓中期】MA20开始渐渐上行，只要不连续2bar低于MA20就保持持仓
            # # 【止盈】RSI > 80 以后，一旦不上涨就止盈，止盈价格为前一bar close
            # if self.rsi1 > 80 and self.ma_down(self.am.close):
            #     self.sell(bar.close_price, abs(self.pos))
            #     self.write_log(f"[stop gain 1] close position at {bar.close_price}")
            #     self.put_event()
            #     return
            
            # # 【止盈】RSI > 86， bar close时止盈
            # if self.rsi1 > 86:
            #     self.sell(bar.close_price, abs(self.pos))
            #     self.write_log(f"[stop gain 2] close position at {bar.close_price}")
            #     self.put_event()
            #     return
            
            # 连续2bar低于ma20
            sell_cond:bool = self.count_pred(am.close < self.ma20_s, 2) >= 2
            if sell_cond:
                self.sell(bar.close_price, abs(self.pos))
                self.write_log(f"close position at {bar.close_price}")
        
        elif self.pos < 0:
            if is_market_close(bar.datetime):
                self.cover(bar.close_price, abs(self.pos))
                self.write_log(f"close position before marekt close")
                self.put_event()
                return
            
            # 【开空单中期】MA20开始渐渐下行，只要不连续2bar高于MA20就保持持仓
            # 【止盈】RSI < 80 以后，一旦不上涨就止盈，止盈价格为前一bar close
            # if self.rsi1 < 20 and self.ma_up(self.am.close):
            #     self.cover(bar.close_price, abs(self.pos))
            #     self.write_log(f"[stop gain 1] close short position at {bar.close_price}")
            #     self.put_event()
            #     return
            
            # # 【止盈】RSI < 15， bar close时止盈
            # if self.rsi1 < 15:
            #     self.cover(bar.close_price, abs(self.pos))
            #     self.write_log(f"[stop gain 2] close short position at {bar.close_price}")
            #     self.put_event()
            #     return
            
            # 连续2bar高于ma20
            sell_cond:bool = self.count_pred(am.close > self.ma20_s, 2) >= 2
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
        return bar.close_price

    def get_open_short_price(self, bar:BarData, ma10:float, ma20:float):
        """
        确定开仓价，考虑回踩MA10
        """
        # 取MA10和MA20中较小者, 四舍五入（向下取整）后，再减op_offset_px
        op_px = math.floor( min(ma10, ma20) ) - self.op_offset_px
        op_px = max(bar.close_price, op_px) # 多单开仓价不小于前收价
        return bar.close_price

    def calc_ma_series(self):
        '''计算需要的均线序列'''
        am = self.am
        
        self.ma5_s = am.sma(self.N1, array=True)
        self.ma10_s = am.sma(self.N2, array=True)
        self.ma20_s = am.sma(self.N3, array=True)
        self.ma30_s = am.sma(self.N4, array=True)
        self.ma60_s = am.sma(self.N5, array=True)
        self.ma120_s = am.sma(self.N6, array=True)
        self.ma250_s = am.sma(self.N7, array=True)
        
        self.vol_ma20_s = talib.SMA(am.volume, 20)
        self.vol_ma60_s = talib.SMA(am.volume, 10)
    
    def calc_vol_rate(self, bar:BarData) -> tuple[float, float]:
        vol_ma20 = self.vol_ma20_s[-1]
        vol_rate_20 = bar.volume / vol_ma20
        vol_ma60 = self.vol_ma60_s[-1]
        vol_rate_60 = bar.volume / vol_ma60
        return vol_rate_20, vol_rate_60

    def calc_diff_sum(self, DIF_M_P:int = 10) -> float:
        '''计算均线距离之和以及移动平均
        MA_DIF1:=MA10/MA20-1;
        MA_DIF2:=MA20/MA30-1;
        MA_DIF3:=MA30/MA60-1;
        MA_DIF4:=MA60/MA120-1;
        MA_DIF5:=MA120/MA250-1;
        DIF_M:=10;
        DIF_SUM:=10000*(MA_DIF1+MA_DIF2+MA_DIF3+MA_DIF4+MA_DIF5);
        DIF_MA:=MA(DIF_SUM, DIF_M);
        '''

        self.ma5 = self.ma5_s[-1]
        self.ma10 = self.ma10_s[-1]
        self.ma20 = self.ma20_s[-1]
        self.ma30 = self.ma30_s[-1]
        self.ma60 = self.ma60_s[-1]
        self.ma120 = self.ma120_s[-1]
        self.ma250 = self.ma250_s[-1]
        
        ma_dif1:np.ndarray = self.ma10_s / self.ma20_s - 1
        ma_dif2:np.ndarray = self.ma20_s / self.ma30_s - 1
        ma_dif3:np.ndarray = self.ma30_s / self.ma60_s - 1
        ma_dif4:np.ndarray = self.ma60_s / self.ma120_s - 1
        ma_dif5:np.ndarray = self.ma120_s / self.ma250_s - 1
        
        dif_sum_s:np.ndarray = 10000 * (ma_dif1 + ma_dif2 + ma_dif3 + ma_dif4 + ma_dif5)
        dif_ma_s:np.ndarray = talib.SMA(dif_sum_s, DIF_M_P)
        self.dif_sum = dif_sum_s[-1]
        self.dif_ma = dif_ma_s[-1]
        
        return self.dif_ma
    
    def calc_bias_diff(self, ma1:np.ndarray, ma2:np.ndarray, ma3:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        close = self.am.close
        bias1 = (close - ma1) / ma1 * 100
        bias2 = (close - ma2) / ma2 * 100
        bias3 = (close - ma3) / ma3 * 100
        avg_bias = (bias1 + bias2 + bias3) / 3
        diff_bias = abs(bias1 - avg_bias) + abs(bias2 - avg_bias) + abs(bias3 - avg_bias)
        return bias1, diff_bias
    
    def calc_cci(self, CCI_M_P:int = 10) -> tuple[float, float]:
        '''计算CCI指标'''
        am = self.am
        cci_s:np.ndarray = am.cci(14, array=True)
        self.cci = cci_s[-1]
        cci_ma_s:np.ndarray = talib.SMA(cci_s, CCI_M_P)
        self.cci_ma = cci_ma_s[-1]
        return self.cci, self.cci_ma
    
    def calc_rsi(self, RSI_P:int = 5) -> float:
        am = self.am
        rsi_val =  am.rsi(RSI_P)
        return rsi_val

    def calc_my_macd(self, MA_SHORT:np.ndarray, MA_LONG:np.ndarray, MID_P:int = 10) -> np.ndarray:
        '''计算定制MACD
        DIF:=100*(MA(CLOSE,SHORT)/MA(CLOSE,LONG)-1);
        DEA:=MA(DIF,MID);
        MACD:=DIF-DEA;
        '''
        dif_s:np.ndarray = 100 * ( MA_SHORT / MA_LONG - 1)
        dea_s:np.ndarray = talib.SMA(dif_s, MID_P)
        macd_s:np.ndarray = dif_s - dea_s
        return macd_s

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
