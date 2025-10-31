import numpy as np
import pandas as pd

class TrendAnalyzer:
    @staticmethod
    def calculate_sma(prices, period):
        return pd.Series(prices).rolling(window=period).mean().tolist()

    @staticmethod
    def calculate_ema(prices, period):
        return pd.Series(prices).ewm(span=period, adjust=False).mean().tolist()

    @staticmethod
    def calculate_momentum(prices, period):
        momentum = np.diff(prices, n=period, prepend=prices[0])
        return [0] * (period - 1) + momentum.tolist()

    @staticmethod
    def calculate_rsi(prices, period=14):
        price_series = pd.Series(prices)
        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).tolist()

    @staticmethod
    def calculate_macd(prices, short_period=12, long_period=26, signal_period=9):
        short_ema = pd.Series(prices).ewm(span=short_period, adjust=False).mean()
        long_ema = pd.Series(prices).ewm(span=long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        return macd.tolist(), signal_line.tolist()
    
    @staticmethod
    def detect_double_top(prices):
        if len(prices) < 5:
            return 0
        return int(prices[-1] < prices[-3] and prices[-2] > prices[-4] and prices[-3] > prices[-5])

    @staticmethod
    def detect_double_bottom(prices):
        if len(prices) < 5:
            return 0
        return int(prices[-1] > prices[-3] and prices[-2] < prices[-4] and prices[-3] < prices[-5])
    
    @staticmethod
    def identify_support_resistance(prices, window=10):
        support = min(prices[-window:])
        resistance = max(prices[-window:])
        return support, resistance
    
    @staticmethod
    def detect_breakout(prices, support, resistance):
        latest_price = prices[-1]
        if latest_price > resistance:
            return 1  # Breakout above resistance (bullish)
        elif latest_price < support:
            return -1  # Breakdown below support (bearish)
        return 0  # No breakout
    
    @staticmethod
    def predict_trend(candles):
        if len(candles) < 21:
            print("⚠️ Not enough data for trend prediction.")
            return -1, 0

        closes = [c[3] for c in candles]  # Extract close prices

        ema_9 = TrendAnalyzer.calculate_ema(closes, 9)
        ema_21 = TrendAnalyzer.calculate_ema(closes, 21)
        macd, macd_signal = TrendAnalyzer.calculate_macd(closes)
        momentum = TrendAnalyzer.calculate_momentum(closes, 3)
        rsi = TrendAnalyzer.calculate_rsi(closes, 7)
        double_top = TrendAnalyzer.detect_double_top(closes)
        double_bottom = TrendAnalyzer.detect_double_bottom(closes)
        support, resistance = TrendAnalyzer.identify_support_resistance(closes)
        breakout = TrendAnalyzer.detect_breakout(closes, support, resistance)

        # Weighted Scoring
        weights = {
            "ema_crossover": 1.5,
            "macd_signal_crossover": 1.2,
            "momentum_positive": 1.0,
            "double_bottom": 1.8,
            "double_top": -1.8,
            "breakout": 2.0
        }

        ema_crossover = int(ema_9[-1] > ema_21[-1]) * weights["ema_crossover"]
        macd_signal_crossover = int(macd[-1] > macd_signal[-1]) * weights["macd_signal_crossover"]
        momentum_positive = int(momentum[-1] > 0) * weights["momentum_positive"]
        double_bottom_score = double_bottom * weights["double_bottom"]
        double_top_score = double_top * weights["double_top"]
        breakout_score = breakout * weights["breakout"]

        trend_score = sum([ema_crossover, macd_signal_crossover, momentum_positive, double_bottom_score, double_top_score, breakout_score])
        max_score = sum(weights.values())
        probability = (trend_score / max_score) * 100

        trend = 1 if probability >= 70 else 0 if probability <= 30 else -1

        # Print trend score and components
        print(f"📊 Trend Score Breakdown:")
        print(f"   EMA Crossover: {ema_crossover}")
        print(f"   MACD Signal Crossover: {macd_signal_crossover}")
        print(f"   Momentum Positive: {momentum_positive}")
        print(f"   Double Bottom: {double_bottom_score}")
        print(f"   Double Top: {double_top_score}")
        print(f"   Breakout: {breakout_score}")
        print(f"➡️  Final Trend Score: {trend_score:.2f} / {max_score:.2f}")
        print(f"📈 Trend Decision: {'Uptrend' if trend == 1 else 'Downtrend' if trend == 0 else 'Uncertain'}")
        print(f"📊 Probability: {probability:.2f}%")

        return trend, probability
