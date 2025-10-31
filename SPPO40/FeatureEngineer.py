import pandas as pd
import ta
import numpy as np


class FeatureEngineer:
    def __init__(self, window=30):
        self.window = window

    def compute_features(self, df):

        df = df.copy()

        # ✅ Dynamically set window to avoid issues with short datasets
        self.window = min(len(df), self.window)

        # Ensure timestamp is in datetime format
        df.loc[:, 'timestamp'] = pd.to_datetime(df['timestamp'])

        # Price Change %
        df.loc[:, 'price_change_pct'] = df['close'].pct_change() * 100

        # Momentum
        df.loc[:, 'momentum_3'] = df['close'].diff(3)
        df.loc[:, 'momentum_5'] = df['close'].diff(5)

        # Moving Averages
        df.loc[:, 'ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df.loc[:, 'ema_21'] = ta.trend.ema_indicator(df['close'], window=21)

        # ATR with dynamic window
        df.loc[:, 'atr_5'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=5)

        # MACD
        df.loc[:, 'macd'] = ta.trend.macd(df['close'])
        df.loc[:, 'macd_signal'] = ta.trend.macd_signal(df['close'])

        # RSI
        df.loc[:, 'rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        df.loc[:, 'bb_upper'] = bb.bollinger_hband()
        df.loc[:, 'bb_lower'] = bb.bollinger_lband()

        # Price Action Strength
        df.loc[:, 'candle_body_size'] = abs(df['close'] - df['open'])
        df.loc[:, 'wick_size'] = df['high'] - df['low']

        # Buy/Sell Pressure
        df.loc[:, 'buy_pressure'] = abs(df['close'] - df['low'])
        df.loc[:, 'sell_pressure'] = abs(df['high'] - df['close'])

        # ✅ Pivot Points (Support/Resistance)
        df.loc[:, 'pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
        df.loc[:, 'support_1'] = (2 * df['pivot_point']) - df['high']
        df.loc[:, 'resistance_1'] = (2 * df['pivot_point']) - df['low']
        df.loc[:, 'support_2'] = df['pivot_point'] - (df['high'] - df['low'])
        df.loc[:, 'resistance_2'] = df['pivot_point'] + (df['high'] - df['low'])

        # ✅ Fibonacci Retracement
        highest_high = df['high'].rolling(window=self.window, min_periods=1).max()
        lowest_low = df['low'].rolling(window=self.window, min_periods=1).min()
        df.loc[:, 'fib_23'] = lowest_low + 0.236 * (highest_high - lowest_low)
        df.loc[:, 'fib_38'] = lowest_low + 0.382 * (highest_high - lowest_low)
        df.loc[:, 'fib_50'] = lowest_low + 0.5 * (highest_high - lowest_low)
        df.loc[:, 'fib_61'] = lowest_low + 0.618 * (highest_high - lowest_low)

        # ✅ Donchian Channel
        df.loc[:, 'donchian_high'] = df['high'].rolling(window=self.window, min_periods=1).max()
        df.loc[:, 'donchian_low'] = df['low'].rolling(window=self.window, min_periods=1).min()
        df.loc[:, 'donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2

        # ✅ Choppiness Index
        high_low_diff = df['high'].rolling(window=self.window, min_periods=1).max() - df['low'].rolling(window=self.window, min_periods=1).min()
        atr_sum = df['atr_5'].rolling(window=self.window, min_periods=1).sum()
        df.loc[:, 'choppiness_index'] = 100 * np.log10(atr_sum / high_low_diff) / np.log10(self.window)

        # ✅ Market Structure Break
        df.loc[:, 'msb_high'] = df['high'].rolling(window=self.window, min_periods=1).max()
        df.loc[:, 'msb_low'] = df['low'].rolling(window=self.window, min_periods=1).min()
        df.loc[:, 'breakout_signal'] = np.where(df['close'] > df['msb_high'], 1, 
                                                np.where(df['close'] < df['msb_low'], -1, 0))

        # ✅ Price Range Expansion/Contraction
        df.loc[:, 'range_expansion'] = df['high'] - df['low']
        df.loc[:, 'range_contraction'] = df['open'] - df['close']

        # ✅ high-low Breakout Strength (Fixed Typo)
        df.loc[:, 'breakout_strength'] = df['close'] - df['donchian_mid']

        # ✅ Fill missing values to prevent NaN
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        return df

    



     
