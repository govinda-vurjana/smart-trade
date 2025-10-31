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


         # ✅ Pivot Points (Support/Resistance)
        df['pivot_point'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['support_1'] = (2 * df['pivot_point']) - df['High']
        df['resistance_1'] = (2 * df['pivot_point']) - df['Low']
        df['support_2'] = df['pivot_point'] - (df['High'] - df['Low'])
        df['resistance_2'] = df['pivot_point'] + (df['High'] - df['Low'])

        # ✅ Fibonacci Retracement (Breakout Points)
        highest_high = df['High'].rolling(window=self.window).max()
        lowest_low = df['Low'].rolling(window=self.window).min()
        df['fib_23'] = lowest_low + 0.236 * (highest_high - lowest_low)
      
        # ✅ Donchian Channel (Breakout Strength)
        df['donchian_high'] = df['High'].rolling(window=self.window).max()
        df['donchian_low'] = df['Low'].rolling(window=self.window).min()
        df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2

        # ✅ Choppiness Index (Trend vs Consolidation)
        high_low_diff = df['High'].rolling(window=self.window).max() - df['Low'].rolling(window=self.window).min()
        atr_sum = df['atr_5'].rolling(window=self.window).sum()

        

        df.loc[:, 'fib_38'] = lowest_low + 0.382 * (highest_high - lowest_low)
        df.loc[:, 'fib_50'] = lowest_low + 0.5 * (highest_high - lowest_low)
        df.loc[:, 'fib_61'] = lowest_low + 0.618 * (highest_high - lowest_low)

        df.loc[:, 'donchian_high'] = df['High'].rolling(window=self.window).max()
        df.loc[:, 'donchian_low'] = df['Low'].rolling(window=self.window).min()
        df.loc[:, 'donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2

        df.loc[:, 'choppiness_index'] = 100 * np.log10(atr_sum / high_low_diff) / np.log10(self.window)

        df.loc[:, 'msb_high'] = df['High'].rolling(window=self.window).max()
        df.loc[:, 'msb_low'] = df['Low'].rolling(window=self.window).min()

        df.loc[:, 'breakout_signal'] = np.where(df['Close'] > df['msb_high'], 1,
                                                np.where(df['Close'] < df['msb_low'], -1, 0))

        df.loc[:, 'range_expansion'] = df['High'] - df['Low']
        df.loc[:, 'range_contraction'] = df['Open'] - df['Close']


        # ✅ High-Low Breakout Strength
        df['breakout_starength'] = df['Close'] - df['donchian_mid']

        # df['head_shoulders'], df['head_shoulders_target'] = self.detect_head_and_shoulders(df)
        # df['double_top'], df['double_top_target'] = self.detect_double_top(df)
        # df['cup_handle'], df['cup_handle_target'] = self.detect_cup_and_handle(df)
        # df['wedge_pattern'], df['wedge_target'] = self.detect_wedge_pattern(df)
        # df['triangle_pattern'], df['triangle_target'] = self.detect_triangle_pattern(df)
        # df['breakaway_gap'], df['breakaway_gap_target'] = self.detect_breakaway_gap(df)
        # df['bull_trap'], df['bull_trap_target'] = self.detect_bull_trap(df)

         # Fill missing values to prevent NaN
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # Return the last 12 rows as state
        return df.iloc[-12:]



     

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




