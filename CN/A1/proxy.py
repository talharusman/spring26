#!/usr/bin/env python3
"""
Simple HTTP/1.0 Proxy Server
- Accepts GET requests from clients, forwards them to remote servers
- Uses threading for concurrent connections (works on Windows & Linux)
- Usage: python proxy.py <port>
"""

import sys
import socket
import threading

BUFFER_SIZE = 4096
TIMEOUT = 10  # seconds


# ── URL Parsing ──────────────────────────────────────────────────────────────

def parse_url(uri):
    """Parse 'http://host[:port]/path' → (host, port, path)."""
    if not uri.startswith("http://"):
        raise ValueError("URI must start with http://")
    rest = uri[7:]                          # strip "http://"
    slash = rest.find("/")
    host_port = rest[:slash] if slash != -1 else rest
    path = rest[slash:] if slash != -1 else "/"
    if ":" in host_port:
        host, port = host_port.split(":", 1)
        port = int(port)
    else:
        host, port = host_port, 80
    if not host:
        raise ValueError("Empty host")
    return host, port, path


# ── Request Parsing ──────────────────────────────────────────────────────────

def parse_request(raw):
    """Parse raw HTTP request bytes → (method, uri, headers dict)."""
    text = raw.decode("utf-8", errors="replace")
    header_end = text.find("\r\n\r\n")
    if header_end == -1:
        header_end = text.find("\n\n")
    header_section = text[:header_end] if header_end != -1 else text
    lines = header_section.replace("\r\n", "\n").split("\n")

    # Request line: "GET http://... HTTP/1.0"
    parts = lines[0].split()
    if len(parts) != 3:
        raise ValueError("400 Malformed request line")
    method, uri, version = parts
    if method != "GET":
        raise ValueError("501 Only GET is supported")
    if not uri.startswith("http://"):
        raise ValueError("400 URI must be absolute (http://...)")

    # Headers
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip()] = v.strip()
    return method, uri, headers


# ── Error Response ───────────────────────────────────────────────────────────

def send_error(sock, code, msg):
    """Send a simple HTTP error response."""
    body = f"<html><body><h1>{code} {msg}</h1></body></html>"
    resp = (f"HTTP/1.0 {code} {msg}\r\n"
            f"Content-Type: text/html\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n\r\n{body}")
    try:
        sock.sendall(resp.encode())
    except Exception:
        pass


# ── Client Handler (runs in a thread) ───────────────────────────────────────

def handle_client(client_sock, addr):
    """Read client request → forward to server → relay response back."""
    try:
        # 1) Read request
        client_sock.settimeout(TIMEOUT)
        data = b""
        while True:
            try:
                chunk = client_sock.recv(BUFFER_SIZE)
            except socket.timeout:
                break
            if not chunk:
                break
            data += chunk
            if b"\r\n\r\n" in data or b"\n\n" in data:
                break

        if not data:
            send_error(client_sock, 400, "Bad Request")
            return

        # 2) Parse request
        try:
            method, uri, headers = parse_request(data)
            host, port, path = parse_url(uri)
        except ValueError as e:
            code = int(str(e)[:3]) if str(e)[:3].isdigit() else 400
            send_error(client_sock, code, str(e)[4:] or "Bad Request")
            return

        # 3) Connect to remote server
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.settimeout(TIMEOUT)
            srv.connect((host, port))
        except Exception:
            send_error(client_sock, 502, "Bad Gateway")
            return

        # 4) Build & send forwarded request
        headers.setdefault("Host", host if port == 80 else f"{host}:{port}")
        headers.pop("Proxy-Connection", None)
        headers["Connection"] = "close"

        req = f"GET {path} HTTP/1.0\r\n"
        req += "".join(f"{k}: {v}\r\n" for k, v in headers.items())
        req += "\r\n"

        try:
            srv.sendall(req.encode())
        except Exception:
            send_error(client_sock, 502, "Bad Gateway")
            srv.close()
            return

        # 5) Relay response back to client
        try:
            while True:
                chunk = srv.recv(BUFFER_SIZE)
                if not chunk:
                    break
                client_sock.sendall(chunk)
        except Exception:
            pass
        finally:
            srv.close()

    except Exception:
        send_error(client_sock, 500, "Internal Server Error")
    finally:
        client_sock.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        print("Usage: python proxy.py <port>")
        sys.exit(1)

    port = int(sys.argv[1])
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("", port))
    server.listen(50)
    print(f"[*] Proxy listening on port {port} ...")

    try:
        while True:
            client_sock, addr = server.accept()
            print(f"[+] Connection from {addr[0]}:{addr[1]}")
            t = threading.Thread(target=handle_client, args=(client_sock, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n[*] Shutting down.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
