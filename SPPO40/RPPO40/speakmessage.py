import pyttsx3
import threading
import pythoncom  # Import pythoncom

def speak_message(message: str):
    pythoncom.CoInitialize()  # Initialize COM for the thread

    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)

    # Speak the message
    engine.say(message)
    engine.runAndWait()

    pythoncom.CoUninitialize()  # Uninitialize COM after use

def speak_in_background(message: str):
    speech_thread = threading.Thread(target=speak_message, args=(message,))
    speech_thread.start()
