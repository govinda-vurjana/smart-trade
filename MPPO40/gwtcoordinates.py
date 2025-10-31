import pyautogui
import time

# # for i in range(7):
# time.sleep(10)
# x, y = pyautogui.position()
# print(f"Position: X={x}, Y={y}")
    
import pyautogui
import time

# Predefined positions for each value
positions = {
    5: (2265, 376),
    10: (2349, 385),
    15: (2443, 378),
    30: (2260, 453),
    60: (2340, 452),
    120: (2447, 459),
}
# ✅ Convert Action to Click
position = {"down": (2336, 690), "up": (2336, 531)}
def click_button(action):
    # click_position(time_period)
    print("Trade click button executed...")
    msg=action
    # speak_in_background(f"Action of model is {msg}")
    if action == 0:
        pyautogui.click(*position['up'])
    elif action == 1:
        pyautogui.click(*position['down'])

def click_position(value,action):
    time.sleep(10)
    if value not in positions:
        print("Invalid value. Please use one of [5, 10, 15, 30, 60, 120].")
        return
    
    # Click on the initial position
    initial_x, initial_y = 2347, 303
    pyautogui.click(initial_x, initial_y)
    time.sleep(1)  # Delay before clicking the second position

    # Click on the corresponding position based on the value
    x, y = positions[value]
    pyautogui.click(x, y)
    print(f"Clicked on initial position ({initial_x}, {initial_y}) and then ({x}, {y}) for value {value}.")
    click_button(action)

# Example usage:
click_position(120)  # Replace with the required value
