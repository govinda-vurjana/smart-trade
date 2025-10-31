import socket

# Constants
TARGET_IP = "192.168.31.110"  # Change this to the server's LAN IP
PORT = 65232

class TCPClient:
    def __init__(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((TARGET_IP, PORT))  # Connect only once

    def send_message(self, message):
        """Send a message using the existing connection."""
        try:
            self.client_socket.sendall(message.encode())  # Send message
            print(f"📤 Sent: {message} to {TARGET_IP}:{PORT}")

        except Exception as e:
            print(f"⚠️ Error: {e}")

    def close_connection(self):
        """Close the connection properly."""
        self.client_socket.close()
        print("🔌 Connection closed.")


