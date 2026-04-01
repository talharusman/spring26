import socket
import threading
import os

# Setup server socket
chat_server = socket.socket()
chat_server.bind(("localhost", 9997))
chat_server.listen(5)
print("Chat server is running and waiting for clients")

active_clients = []
permitted_files = [".txt", ".jpg", ".pdf"]
blocked_terms = ["badword", "ugly", "hate"]


def handle_client_connection(conn, addr):
    print("Client connected from:", addr)
    active_clients.append(conn)

    while True:
        try:
            incoming_bytes = conn.recv(4096)

            if not incoming_bytes:
                break

            text_data = incoming_bytes.decode(errors="ignore")

            # Handle text messages
            if text_data.startswith("MSG:"):
                clean_message = text_data[4:]

                if any(term in clean_message.lower() for term in blocked_terms):
                    conn.send(
                        "Message is blocked due to restricted language.".encode()
                    )
                    continue

                print(f"Message comes from {addr}: {clean_message}")

                for client in active_clients:
                    if client != conn:
                        client.send(
                            f"Message comes from {addr}: {clean_message}".encode()
                        )

            # Handle file transfer
            elif text_data.startswith("FILE:"):
                file_name = text_data[5:].strip()
                file_ext = os.path.splitext(file_name)[1]

                if file_ext not in permitted_files:
                    conn.send(
                        "File is rejected. Allowed formats: .txt, .jpg, .pdf".encode()
                    )
                    continue

                conn.send("File is approved. Receiving now...".encode())

                file_content = conn.recv(4096)

                save_name = "received_" + os.path.basename(file_name)
                with open(save_name, "wb") as f:
                    f.write(file_content)

                print(f"Received file from {addr}: {file_name}")

                for client in active_clients:
                    if client != conn:
                        client.send(
                            f"File received from {addr}: {file_name}".encode()
                        )

        except:
            break

    if conn in active_clients:
        active_clients.remove(conn)

    conn.close()
    print("Client connection closed:", addr)


# Accept incoming connections
while True:
    new_conn, new_addr = chat_server.accept()
    threading.Thread(
        target=handle_client_connection,
        args=(new_conn, new_addr)
    ).start()
