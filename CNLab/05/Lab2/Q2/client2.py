import socket
import threading

# Create client socket and connect to server
chat_client = socket.socket()
chat_client.connect(("localhost", 9997))
print("Successfully connected to the chat server")

def receive_from_server():
    while True:
        try:
            server_msg = chat_client.recv(1024).decode()

            if not server_msg or server_msg.lower() == "exit":
                print("Server has closed the connection")
                chat_client.close()
                break

            print("Server:", server_msg)

        except:
            break

# Start a separate thread to handle incoming messages
receive_thread = threading.Thread(target=receive_from_server)
receive_thread.start()

# Send messages to server
while True:
    client_msg = input("Client: ")
    chat_client.send(client_msg.encode())

    if client_msg.lower() == "exit":
        print("Client terminated the connection")
        chat_client.close()
        break
