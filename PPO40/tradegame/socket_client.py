import socket
import time

class StatusLogger:
    def __init__(self, server_ip, port):
        self.server_ip = server_ip
        self.port = port
        self.client = None

    def connect(self):
        while True:
            try:
                # ✅ Create a new socket connection
                self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client.connect((self.server_ip, self.port))
                print("✅ Connected to server.")
                break
            except Exception as e:
                print("❌ Connection failed. Retrying in 5 seconds...")
                time.sleep(5)

    def log_message(self, message):
        if self.client is None:
            self.connect()

        # ✅ Send the log message
        try:
            self.client.send(message.encode())
            print(f"✅ Sent: {message}")
        except Exception as e:
            print("❌ Connection lost. Reconnecting...")
            self.client.close()
            self.connect()

# ✅ Initialize the socket logger
logger = StatusLogger(server_ip='203.122.43.21', port=65432)

# ✅ Example of sending unstructured log messages
while True:
    logger.log_message("✅ Trade Triggered - BUY @ 102.21")
    time.sleep(5)

    logger.log_message("❌ Stop Loss Hit - LOSS ₹500")
    time.sleep(5)

    logger.log_message("✅ Profit Booked - ₹1020")
    time.sleep(5)

    logger.log_message("⚠️ Warning: Connection unstable.")
    time.sleep(5)
