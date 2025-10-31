import pandas as pd
import ta
import numpy as np


class FeatureEngineer:
    def __init__(self, window=30):
        self.window = window

    def compute_features(self, df):
        # Ensure Timestamp is in datetime format
        df['Timestamp (IST)'] = pd.to_datetime(df['Timestamp (IST)'])

        # Price Change %
        df['price_change_pct'] = df['Close'].pct_change() * 100

        # Momentum
        df['momentum_3'] = df['Close'].diff(3)
        df['momentum_5'] = df['Close'].diff(5)

        # Moving Averages (within 30-period max)
        df['ema_9'] = ta.trend.ema_indicator(df['Close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['Close'], window=21)

        # Average True Range (ATR)
        df['atr_5'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=5)

        # MACD
        df['macd'] = ta.trend.macd(df['Close'])
        df['macd_signal'] = ta.trend.macd_signal(df['Close'])

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()

        # Price Action Strength
        df['candle_body_size'] = abs(df['Close'] - df['Open'])
        df['wick_size'] = df['High'] - df['Low']

        # Buy/Sell Pressure
        df['buy_pressure'] = abs(df['Close'] - df['Low'])
        df['sell_pressure'] = abs(df['High'] - df['Close'])

        # ✅ Remove any future-based columns (no future_close, no future_change)


        #  # ✅ Pivot Points (Support/Resistance)
        # df['pivot_point'] = (df['High'] + df['Low'] + df['Close']) / 3
        # df['support_1'] = (2 * df['pivot_point']) - df['High']
        # df['resistance_1'] = (2 * df['pivot_point']) - df['Low']
        # df['support_2'] = df['pivot_point'] - (df['High'] - df['Low'])
        # df['resistance_2'] = df['pivot_point'] + (df['High'] - df['Low'])

        # # ✅ Fibonacci Retracement (Breakout Points)
        # highest_high = df['High'].rolling(window=self.window).max()
        # lowest_low = df['Low'].rolling(window=self.window).min()
        # df['fib_23'] = lowest_low + 0.236 * (highest_high - lowest_low)
        # df['fib_38'] = lowest_low + 0.382 * (highest_high - lowest_low)
        # df['fib_50'] = lowest_low + 0.5 * (highest_high - lowest_low)
        # df['fib_61'] = lowest_low + 0.618 * (highest_high - lowest_low)

        # # ✅ Donchian Channel (Breakout Strength)
        # df['donchian_high'] = df['High'].rolling(window=self.window).max()
        # df['donchian_low'] = df['Low'].rolling(window=self.window).min()
        # df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2

        # # ✅ Choppiness Index (Trend vs Consolidation)
        # high_low_diff = df['High'].rolling(window=self.window).max() - df['Low'].rolling(window=self.window).min()
        # atr_sum = df['atr_5'].rolling(window=self.window).sum()
        # df['choppiness_index'] = 100 * np.log10(atr_sum / high_low_diff) / np.log10(self.window)

        # # ✅ Market Structure Break (MSB) (Strong Breakouts)
        # df['msb_high'] = df['High'].rolling(window=self.window).max()
        # df['msb_low'] = df['Low'].rolling(window=self.window).min()
        # df['breakout_signal'] = np.where(df['Close'] > df['msb_high'], 1, 
        #                                 np.where(df['Close'] < df['msb_low'], -1, 0))

        # # ✅ Price Range Expansion/Contraction
        # df['range_expansion'] = df['High'] - df['Low']
        # df['range_contraction'] = df['Open'] - df['Close']

        # # ✅ High-Low Breakout Strength
        # df['breakout_starength'] = df['Close'] - df['donchian_mid']


        # Fill missing values to prevent NaN
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # Return the last 12 rows as state
        return df.iloc[-12:]




