import socket

def run_client():
    # Create client socket and connect to server
    client = socket.socket()
    client.connect(("localhost", 9998))
    print("Successfully connected to the server")

    while True:
        choice = input("Send message (M) or file (F)? ").strip()
        client.send(choice.encode())

        
        if choice.lower() == "m":# Message sending mode
            msg = input("Client: ")
            client.send(f"MSG: {msg}".encode())

            if msg.lower() == "exit":
                print("Connection closed by client")
                client.close()
                break

            reply = client.recv(1024).decode()

            if not reply or reply.lower() == "exit":
                print("Server terminated the connection")
                client.close()
                break

            print("Server:", reply)

        # File sending mode
        elif choice.lower() == "f":
            filename = input("Enter file name: ")
            client.send(f"FILE: {filename}".encode())

            try:
                with open(filename, "rb") as f:
                    file_bytes = f.read()
                    client.send(file_bytes)
                print("File transferred successfully")

            except FileNotFoundError:
                print("Error: File does not exist")
                continue

            reply = client.recv(1024).decode()
            print("Server:", reply)

            if reply.lower() == "exit":
                print("Server closed the connection")
                client.close()
                break

        else:
            print("Invalid choice. Please select M or F.")

run_client()
