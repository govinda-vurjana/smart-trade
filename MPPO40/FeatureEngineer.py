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

        # Ensure Timestamp is in datetime format
        df.loc[:, 'Timestamp (IST)'] = pd.to_datetime(df['Timestamp (IST)'])

        # Price Change %
        df.loc[:, 'price_change_pct'] = df['Close'].pct_change() * 100

        # Momentum
        df.loc[:, 'momentum_3'] = df['Close'].diff(3)
        df.loc[:, 'momentum_5'] = df['Close'].diff(5)

        # Moving Averages
        df.loc[:, 'ema_9'] = ta.trend.ema_indicator(df['Close'], window=9)
        df.loc[:, 'ema_21'] = ta.trend.ema_indicator(df['Close'], window=21)

        # ATR with dynamic window
        df.loc[:, 'atr_5'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=5)

        # MACD
        df.loc[:, 'macd'] = ta.trend.macd(df['Close'])
        df.loc[:, 'macd_signal'] = ta.trend.macd_signal(df['Close'])

        # RSI
        df.loc[:, 'rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20)
        df.loc[:, 'bb_upper'] = bb.bollinger_hband()
        df.loc[:, 'bb_lower'] = bb.bollinger_lband()

        # Price Action Strength
        df.loc[:, 'candle_body_size'] = abs(df['Close'] - df['Open'])
        df.loc[:, 'wick_size'] = df['High'] - df['Low']

        # Buy/Sell Pressure
        df.loc[:, 'buy_pressure'] = abs(df['Close'] - df['Low'])
        df.loc[:, 'sell_pressure'] = abs(df['High'] - df['Close'])

        # ✅ Pivot Points (Support/Resistance)
        df.loc[:, 'pivot_point'] = (df['High'] + df['Low'] + df['Close']) / 3
        df.loc[:, 'support_1'] = (2 * df['pivot_point']) - df['High']
        df.loc[:, 'resistance_1'] = (2 * df['pivot_point']) - df['Low']
        df.loc[:, 'support_2'] = df['pivot_point'] - (df['High'] - df['Low'])
        df.loc[:, 'resistance_2'] = df['pivot_point'] + (df['High'] - df['Low'])

        # ✅ Fibonacci Retracement
        highest_high = df['High'].rolling(window=self.window, min_periods=1).max()
        lowest_low = df['Low'].rolling(window=self.window, min_periods=1).min()
        df.loc[:, 'fib_23'] = lowest_low + 0.236 * (highest_high - lowest_low)
        df.loc[:, 'fib_38'] = lowest_low + 0.382 * (highest_high - lowest_low)
        df.loc[:, 'fib_50'] = lowest_low + 0.5 * (highest_high - lowest_low)
        df.loc[:, 'fib_61'] = lowest_low + 0.618 * (highest_high - lowest_low)

        # ✅ Donchian Channel
        df.loc[:, 'donchian_high'] = df['High'].rolling(window=self.window, min_periods=1).max()
        df.loc[:, 'donchian_low'] = df['Low'].rolling(window=self.window, min_periods=1).min()
        df.loc[:, 'donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2

        # ✅ Choppiness Index
        high_low_diff = df['High'].rolling(window=self.window, min_periods=1).max() - df['Low'].rolling(window=self.window, min_periods=1).min()
        atr_sum = df['atr_5'].rolling(window=self.window, min_periods=1).sum()
        df.loc[:, 'choppiness_index'] = 100 * np.log10(atr_sum / high_low_diff) / np.log10(self.window)

        # ✅ Market Structure Break
        df.loc[:, 'msb_high'] = df['High'].rolling(window=self.window, min_periods=1).max()
        df.loc[:, 'msb_low'] = df['Low'].rolling(window=self.window, min_periods=1).min()
        df.loc[:, 'breakout_signal'] = np.where(df['Close'] > df['msb_high'], 1, 
                                                np.where(df['Close'] < df['msb_low'], -1, 0))

        # ✅ Price Range Expansion/Contraction
        df.loc[:, 'range_expansion'] = df['High'] - df['Low']
        df.loc[:, 'range_contraction'] = df['Open'] - df['Close']

        # ✅ High-Low Breakout Strength (Fixed Typo)
        df.loc[:, 'breakout_strength'] = df['Close'] - df['donchian_mid']

        # ✅ Fill missing values to prevent NaN
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # ✅ Return the entire DataFrame (not just last 36 rows)
        return df

    



     

    def detect_head_and_shoulders(self, df):
        # ✅ Detect Head & Shoulders + Predict Target
        result = np.zeros(len(df))
        target = np.zeros(len(df))
        for i in range(2, len(df)-2):
            left_shoulder = df['High'].iloc[i-2]
            head = df['High'].iloc[i]
            right_shoulder = df['High'].iloc[i+2]
            
            if left_shoulder < head and right_shoulder < head and abs(left_shoulder - right_shoulder) < 0.02:
                # ✅ Pattern Found
                result[i] = 1
                height = head - left_shoulder
                target[i] = df['Close'].iloc[i+2] - height

        return result, target

    def detect_double_top(self, df):
        # ✅ Detect Double Top + Predict Target
        result = np.zeros(len(df))
        target = np.zeros(len(df))
        for i in range(1, len(df)-2):
            if (df['High'].iloc[i] == df['High'].iloc[i+2]) and (df['High'].iloc[i] > df['High'].iloc[i+1]):
                # ✅ Double Top Found
                result[i] = 1
                height = df['High'].iloc[i] - df['Low'].iloc[i+1]
                target[i] = df['Close'].iloc[i+2] - height

        return result, target

    def detect_cup_and_handle(self, df):
        # ✅ Detect Cup & Handle + Predict Target
        result = np.zeros(len(df))
        target = np.zeros(len(df))
        for i in range(5, len(df)):
            cup = df['Low'].iloc[i-5:i].min()
            handle = df['Low'].iloc[i-2:i].max()
            if handle > cup and (df['Close'].iloc[i] > handle):
                result[i] = 1
                height = handle - cup
                target[i] = df['Close'].iloc[i] + height

        return result, target

    def detect_wedge_pattern(self, df):
        # ✅ Detect Rising/Falling Wedge + Predict Target
        result = np.zeros(len(df))
        target = np.zeros(len(df))
        for i in range(5, len(df)):
            highs = df['High'].iloc[i-5:i]
            lows = df['Low'].iloc[i-5:i]
            if highs.max() - highs.min() < 0.5 * (lows.max() - lows.min()):
                result[i] = 1
                height = highs.max() - lows.min()
                target[i] = df['Close'].iloc[i] - height

        return result, target

    def detect_triangle_pattern(self, df):
        # ✅ Detect Ascending/Descending Triangle + Predict Target
        result = np.zeros(len(df))
        target = np.zeros(len(df))
        for i in range(5, len(df)):
            highs = df['High'].iloc[i-5:i]
            lows = df['Low'].iloc[i-5:i]
            if highs.max() - highs.min() < 0.2 * lows.max():
                result[i] = 1
                height = highs.max() - lows.min()
                target[i] = df['Close'].iloc[i] + height

        return result, target

    def detect_breakaway_gap(self, df):
        # ✅ Detect Breakaway Gap + Predict Target
        result = np.zeros(len(df))
        target = np.zeros(len(df))
        for i in range(1, len(df)):
            if abs(df['Open'].iloc[i] - df['Close'].iloc[i-1]) > 2 * np.std(df['Close'].iloc[i-5:i]):
                result[i] = 1
                height = abs(df['Open'].iloc[i] - df['Close'].iloc[i-1])
                target[i] = df['Close'].iloc[i] + height

        return result, target

    def detect_bull_trap(self, df):
        # ✅ Detect Bull Trap + Predict Target
        result = np.zeros(len(df))
        target = np.zeros(len(df))
        for i in range(3, len(df)):
            if (df['High'].iloc[i] > df['High'].iloc[i-1]) and (df['Close'].iloc[i] < df['Open'].iloc[i]):
                result[i] = 1
                height = df['High'].iloc[i] - df['Low'].iloc[i-1]
                target[i] = df['Close'].iloc[i] - height

        return result, target


        # Fill missing values to prevent NaN
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # Return the last 12 rows as state
        return df.iloc[-12:]




