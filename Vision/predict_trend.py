import numpy as np
import pandas as pd

def is_breakout(df, support_levels, resistance_levels):
    """Detects if the latest candle breaks out of support or resistance."""
    last_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    last_color = df["color"].iloc[-1]

    if last_close > max(resistance_levels) and prev_close <= max(resistance_levels):
        # Bullish breakout with a green candle
        if last_color == "green":
            return 1, 90.0  # Bullish breakout, high confidence
        return 1, 70.0  # Bullish breakout, lower confidence
    elif last_close < min(support_levels) and prev_close >= min(support_levels):
        # Bearish breakout with a red candle
        if last_color == "red":
            return 0, 90.0  # Bearish breakout, high confidence
        return 0, 70.0  # Bearish breakout, lower confidence
    return -1, 50.0  # No breakout

def detect_reversal_pattern(df):
    """Detects common reversal candlestick patterns."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    last_color = last["color"]

    # Hammer (Bullish reversal)
    if last["Low"] < prev["Low"] and (last["Close"] - last["Low"]) > (last["High"] - last["Close"]) * 2:
        if last_color == "green":
            return 1, 80.0  # Bullish reversal with high confidence
        return 1, 60.0  # Bullish reversal with lower confidence

    # Shooting Star (Bearish reversal)
    if last["High"] > prev["High"] and (last["High"] - last["Close"]) > (last["Close"] - last["Low"]) * 2:
        if last_color == "red":
            return 0, 80.0  # Bearish reversal with high confidence
        return 0, 60.0  # Bearish reversal with lower confidence

    return -1, 50.0  # No strong pattern

def predict_trend(candles, support_levels, resistance_levels):
    df = pd.DataFrame(candles, columns=["Index", "Open", "High", "Close", "Low", "x", "y", "width", "height", "color"])

    # Moving Averages
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()

    # Trend direction using Moving Averages
    if df["SMA_5"].iloc[-1] > df["SMA_10"].iloc[-1]:
        moving_avg_signal = 1  # Uptrend
    elif df["SMA_5"].iloc[-1] < df["SMA_10"].iloc[-1]:
        moving_avg_signal = 0  # Downtrend
    else:
        moving_avg_signal = -1  # Neutral

    # Support & Resistance check
    last_close = df["Close"].iloc[-1]
    near_support = any(abs(last_close - level) < (last_close * 0.01) for level in support_levels)
    near_resistance = any(abs(last_close - level) < (last_close * 0.01) for level in resistance_levels)

    if near_support:
        sr_signal = 1  # Likely to bounce up
    elif near_resistance:
        sr_signal = 0  # Likely to fall
    else:
        sr_signal = -1  # No strong SR signal

    # Breakout detection
    breakout_signal, breakout_confidence = is_breakout(df, support_levels, resistance_levels)

    # Reversal pattern detection
    reversal_signal, reversal_confidence = detect_reversal_pattern(df)

    # Decision Logic
    if breakout_signal != -1:
        return breakout_signal, breakout_confidence  # High confidence in breakout
    if reversal_signal != -1:
        return reversal_signal, reversal_confidence  # Reversal pattern detected
    if moving_avg_signal == 1 and sr_signal != 0:
        return 1, 85.0  # Uptrend with confidence
    if moving_avg_signal == 0 and sr_signal != 1:
        return 0, 85.0  # Downtrend with confidence
    if sr_signal != -1:
        return sr_signal, 75.0  # Use SR signal if clear

    return -1, 50.0  # Unclear trend

