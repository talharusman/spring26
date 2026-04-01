import socket
import threading
import os

# Create client socket and connect to server
chat_client = socket.socket()
chat_client.connect(("localhost", 9997))
print("Client connected to the chat server")

def receive_messages():
    while True:
        try:
            server_data = chat_client.recv(4096).decode()

            if not server_data or server_data.lower() == "exit":
                print("Connection closed by server")
                chat_client.close()
                break

            print(server_data)

        except:
            break

# Start thread to receive messages/files from server
listener_thread = threading.Thread(target=receive_messages)
listener_thread.start()

while True:
    user_option = input("Choose message (m) or file (f): ").strip().lower()

    # Send text message
    if user_option == "m":
        text = input("Client: ")
        chat_client.send(f"MSG:{text}".encode())

        if text.lower() == "exit":
            print("Client disconnected")
            chat_client.close()
            break

    # Send file
    elif user_option == "f":
        file_path = input("Enter file name: ")

        if not os.path.isfile(file_path):
            print("Error: File not found")
            continue

        chat_client.send(f"FILE:{file_path}".encode())

        with open(file_path, "rb") as f:
            chat_client.send(f.read())

    else:
        print("Invalid selection, please choose m or f")
