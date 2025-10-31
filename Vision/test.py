import os
import time
import pyautogui
import cv2
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from predict_trend import predict_trend


# Define directories
save_path = "dataset/"
screenshot_dir = os.path.join(save_path, "screenshots")
data_dir = os.path.join(save_path, "data")

os.makedirs(screenshot_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Screenshot region
top_left = (237, 338)
bottom_right = (2138, 1279)
width, height = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]

print("✅ Directories set up.")



# Capture screenshot function
def capture_screenshot():
    timestamp = time.strftime("%H%M%S")
    screenshot_path = os.path.join(screenshot_dir, f"{timestamp}.png")
    print("📸 Capturing screenshot...")
    
    screenshot = pyautogui.screenshot(region=(top_left[0], top_left[1], width, height))
    screenshot.save(screenshot_path)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    print(f"✅ Screenshot saved: {screenshot_path}")
    return screenshot, screenshot_path

import cv2
import numpy as np

def process_image(image):
    print("🔍 Processing image...")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and green candles
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 100, 100]), np.array([180, 255, 255])
    lower_green, upper_green = np.array([35, 100, 100]), np.array([85, 255, 255])

    # Create masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    candlestick_mask = cv2.bitwise_or(mask_red, mask_green)

    # Morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    candlestick_mask = cv2.morphologyEx(candlestick_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(candlestick_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = sorted([cv2.boundingRect(cnt) for cnt in contours], key=lambda b: b[0])

    # Extract OHLC data
    all_candles_data = []

    for i, (x, y, w, h) in enumerate(bounding_boxes):
        high = y  # Top of the candle
        low = y + h  # Bottom of the candle

        roi = hsv[y:y+h, x:x+w]

        # Identify candle color
        hue_values = roi[..., 0].flatten()
        red_ratio = np.sum((hue_values < 10) | (hue_values > 170)) / len(hue_values)
        green_ratio = np.sum((hue_values > 35) & (hue_values < 85)) / len(hue_values)

        color = "Green" if green_ratio > red_ratio else "Red"

        # Identify the body (thicker region) and wicks
        width_profile = np.sum(roi[..., 1] > 50, axis=1)  # Saturation-based width profile
        body_top = np.argmax(width_profile > (0.6 * max(width_profile))) + y
        body_bottom = len(width_profile) - np.argmax(width_profile[::-1] > (0.6 * max(width_profile))) + y - 1

        # Wick calculations
        upper_wick_height = abs(body_top - high)
        lower_wick_height = abs(body_bottom - low)

        # Assign OHLC values based on candle color
        if color == "Green":
            open_price = low + lower_wick_height
            close_price = high - upper_wick_height
        else:  # Red Candle
            open_price = high - upper_wick_height
            close_price = low + lower_wick_height

        all_candles_data.append([i, open_price, high, close_price, low, upper_wick_height, lower_wick_height, x, y, w, h, color])

    print("✅ Candlestick data corrected with accurate OHLC and wick separation.")
    return all_candles_data


from scipy.signal import argrelextrema
def identify_support_resistance(candles, sensitivity=5):
    df = pd.DataFrame(candles, columns=["Index", "Open", "High", "Close", "Low","upper_wick_height", "lower_wick_height",   "x", "y", "width", "height", "color"])
    
    highs = df["High"].tolist()
    lows = df["Low"].tolist()
    
    resistance_levels = []
    support_levels = []
    
    for i in range(sensitivity, len(df) - sensitivity):
        high = highs[i]
        low = lows[i]
        
        if all(high > highs[j] for j in range(i - sensitivity, i + sensitivity + 1) if j != i):
            resistance_levels.append(high)
        
        if all(low < lows[j] for j in range(i - sensitivity, i + sensitivity + 1) if j != i):
            support_levels.append(low)
    
    return sorted(set(resistance_levels), reverse=True), sorted(set(support_levels))


def plot_candlesticks_with_sr(candles, image_path):
    df = pd.DataFrame(candles, columns=["Index", "Open", "High", "Close", "Low","upper_wick_height", "lower_wick_height",    "x", "y", "width", "height", "color"])
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig = px.imshow(image_rgb)

    # Plot candles
    for _, row in df.iterrows():
        box_color = "red" if row["color"] == "Red" else "green"
        fig.add_shape(
            type="rect",
            x0=row["x"], x1=row["x"] + row["width"],
            y0=row["y"], y1=row["y"] + row["height"],
            line=dict(color=box_color, width=2),
        )
    
    # Identify Support & Resistance levels
    resistance_levels, support_levels = identify_support_resistance(candles)

    # Draw horizontal lines for resistances (red) and supports (green)
    for res in resistance_levels:
        fig.add_shape(
            type="line",
            x0=0, x1=image.shape[1],
            y0=res, y1=res,
            line=dict(color="red", width=2, dash="dash")
        )

    for sup in support_levels:
        fig.add_shape(
            type="line",
            x0=0, x1=image.shape[1],
            y0=sup, y1=sup,
            line=dict(color="green", width=2, dash="dash")
        )

    fig.update_layout(title="📊 Candlestick Chart with Support & Resistance", hovermode="closest")
    fig.show()


# Call function with your processed candlestick data


# Automated click function
def click_button(action):
    positions = {"up": (2336, 690), "down": (2336, 531)}
    if action in positions:
        pyautogui.click(*positions[action])
        print(f"✅ Clicked {action} button.")


# Main execution
time.sleep(5)
screenshot, screenshot_path = capture_screenshot()
candles = process_image(screenshot)
print(candles)

if candles:
    # plot_candlesticks_with_sr(candles, screenshot_path)
    resistance_levels, support_levels = identify_support_resistance(candles)

    # trend, confidence = predict_trend(candles, support_levels, resistance_levels)
    # print(f"Trend: {'Up' if trend == 1 else 'Down' if trend == 0 else 'Neutral'}, Confidence: {confidence}%")
    
    # trend, probability = predict_trend(candles)

    # if trend in [0, 1]:
    #     action = "up" if trend == 1 else "down"
    #     if confidence>=80:
    #         click_button(action)
    #     else:
    #         print("Confidence Low")
    #     print(f"🧐 Decision: {action.capitalize()} (Probability: {confidence:.2f}%)")
    # else:
    #     print("⚠️ Unable to determine trend.")
    plot_candlesticks_with_sr(candles, screenshot_path)
