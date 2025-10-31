import pyautogui
import time
import pyperclip  # Required for clipboard operations

def autoPriceUpdate(value):
    """
    Clicks on the specified (x, y) coordinates, selects the existing content,
    and pastes the given number value to replace it.
    
    Args:
        x (int): X-coordinate of the target location.
        y (int): Y-coordinate of the target location.
        value (int/float/str): The number to be pasted.
    """
    x=2309
    y=418
    pyautogui.click(x, y)  # Click on the given coordinates
    time.sleep(0.2)  # Short delay to ensure the field is active
    pyautogui.hotkey('ctrl', 'a')  # Select all existing content
    pyperclip.copy(str(value))  # Copy the new value to clipboard
    pyautogui.hotkey('ctrl', 'v')  # Paste the new value

# Example Usage:
time.sleep(10)
autoPriceUpdate( "1000")
