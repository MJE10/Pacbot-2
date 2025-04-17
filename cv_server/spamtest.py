import socket
import time

# Set up the target address and port
UDP_IP = "127.0.0.1"
UDP_PORT = 7777
MESSAGE = b'\x01\x02'  # 2-byte message

# Create the UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Sending UDP packets to {UDP_IP}:{UDP_PORT} every second. Press Ctrl+C to stop.")

try:
    while True:
        sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
        print(f"Sent: {MESSAGE}")
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    sock.close()
