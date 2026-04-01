import socket
import threading

# Initialize server socket
chat_server = socket.socket()
print("Server socket initialized")

chat_server.bind(("localhost", 9997))
chat_server.listen(5)
print("Server is listening for incoming clients")

active_connections = []

def handle_client(client_socket, client_address):
    print("New client connected:", client_address)
    active_connections.append(client_socket)

    while True:
        try:
            message = client_socket.recv(1024).decode()

            if not message or message.lower() == "exit":
                print("Client left:", client_address)
                active_connections.remove(client_socket)
                client_socket.close()
                break

            print(f"Message from {client_address}:", message)

            # Broadcast message to other connected clients
            for conn in active_connections:
                if conn != client_socket:
                    conn.send(
                        f"From {client_address}: {message}".encode()
                    )

        except:
            if client_socket in active_connections:
                active_connections.remove(client_socket)
            client_socket.close()
            break

# Accept clients continuously
while True:
    client, address = chat_server.accept()
    client_thread = threading.Thread(
        target=handle_client,
        args=(client, address)
    )
    client_thread.start()
