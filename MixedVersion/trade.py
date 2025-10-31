import numpy as np
from scipy.special import softmax

def detect_trend(hh, hl, supports, resistances, live_tick):
    """
    Determines the market trend in real-time based on higher highs (hh), 
    higher lows (hl), support, resistance levels, and live tick data.

    Parameters:
    hh (list of tuples): Higher Highs [(timestamp, price), ...]
    hl (list of tuples): Higher Lows [(timestamp, price), ...]
    supports (list of tuples): Support levels [(timestamp, price), ...]
    resistances (list of tuples): Resistance levels [(timestamp, price), ...]
    live_tick (tuple): Live market price (timestamp, price)

    Returns:
    dict: {'trend': 'Up' or 'Down', 'probability': float}
    """

    if len(hh) < 2 or len(hl) < 2:
        return {"trend": "Neutral", "probability": 0.5}

    # Extract price values
    hh_prices = [price for _, price in hh]
    hl_prices = [price for _, price in hl]
    support_prices = [price for _, price in supports]
    resistance_prices = [price for _, price in resistances]
    
    live_price = live_tick[1]  # Latest live price

    # Trend Strength Calculation
    up_strength = 0
    down_strength = 0

    # Check if live price is breaking resistance (bullish signal)
    if any(live_price > res for res in resistance_prices):
        up_strength += 1.5
    
    # Check if live price is breaking support (bearish signal)
    if any(live_price < sup for sup in support_prices):
        down_strength += 1.5

    # Higher Highs and Higher Lows pattern (bullish)
    if hh_prices[-1] > hh_prices[-2] and hl_prices[-1] > hl_prices[-2]:
        up_strength += 1
    
    # Lower Highs and Lower Lows pattern (bearish)
    if hh_prices[-1] < hh_prices[-2] and hl_prices[-1] < hl_prices[-2]:
        down_strength += 1

    # Use softmax to get probabilities
    probabilities = softmax([down_strength, up_strength])

    trend = "Up" if probabilities[1] > probabilities[0] else "Down"

    return {"trend": trend, "probability": round(float(max(probabilities)), 2)}

