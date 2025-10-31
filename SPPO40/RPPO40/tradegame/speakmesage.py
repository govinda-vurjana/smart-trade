import pyttsx3
import threading

def speak_message(message: str):
    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)

    # Pass the message to be spoken
    engine.say(message)

    # Wait for the speaking to be done
    engine.runAndWait()

def speak_in_background(message: str):
    # Create a new thread for the speech function
    speech_thread = threading.Thread(target=speak_message, args=(message,))
    speech_thread.start()
    

