import socket

def run_server():
    # Create server socket
    server = socket.socket()
    print("Server socket initialized")

    # Bind socket to localhost and port
    server.bind(("localhost", 9998))
    server.listen(5)
    print("Waiting for client connection...")

    while True:
        client_socket, client_address = server.accept()
        print("Client connected from:", client_address)

        while True:
            received_data = client_socket.recv(1024).decode()

            if received_data.startswith("MSG:"):
                MSG = received_data[4:]  # Remove "MSG:"

                if not MSG or MSG.lower() == "exit":
                    print("Client has disconnected")
                    client_socket.close()
                    break

                print("Client says:", MSG)

                server_reply = input("Server: ")
                client_socket.send(server_reply.encode())

                if server_reply.lower() == "exit":
                    print("Server closed the connection")
                    client_socket.close()
                    break

            # Handle file transfer
            elif received_data.startswith("FILE:"):
                filename = received_data[5:]  # Remove "FILE:"
                print("Receiving file:", filename)

                file_content = client_socket.recv(4096)

                with open(filename, "wb") as f:
                    f.write(file_content)

                server_reply = input("Server: ")
                client_socket.send(server_reply.encode())

                if server_reply.lower() == "exit":
                    print("Server closed connection with:", client_address)
                    client_socket.close()
                    break

run_server()
