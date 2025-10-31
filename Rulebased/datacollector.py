import cv2
import numpy as np
import pytesseract
import mss
import time
import csv
import keyboard
from datetime import datetime, timedelta


# Create filename

filename = "live_data.csv" #holds all time data


# Define screen capture area
capture_area = {"top": 128, "left": 2046, "width": 122, "height": 1389}

# Initialize Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_number(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789.')
    return text.strip()

print("Starting in 10 seconds... Get ready!")
time.sleep(10)
print("Recording started! Press 'q' to quit.")

with open(filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Value"])
    
    with mss.mss() as sct:
        start_time = time.time()
        batch_data = []
        
        while True:
            screenshot = sct.grab(capture_area)
            frame = np.array(screenshot)
            value = extract_number(frame)
            
            # Convert UTC to IST
            utc_time = datetime.utcnow()
            ist_time = utc_time + timedelta(hours=5, minutes=30)
            ist_timestamp = ist_time.isoformat()

            if value:
                batch_data.append((ist_timestamp, value))
                writer.writerow([ist_timestamp, value])
                file.flush()

            if time.time() - start_time >= 5:
                if batch_data:
                    print("Batch Data (Last 5 sec):", batch_data)
                batch_data = []  
                start_time = time.time()
            
            if keyboard.is_pressed("q"):
                print("\nQuitting... Saving data!")
                break
            
            time.sleep(1 / 30)

print("Recording stopped. Data saved.")
