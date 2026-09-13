"""Static file server for the live-camera HLS segments (port 8554).

Threaded and pipe-tolerant: probing clients that disconnect mid-transfer
never kill the stream. Serves /tmp/nyc_hls by default.

    .venv/bin/python tools/nyc_cam_serve.py [port] [directory]
"""
import functools
import http.server
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8554
DIRECTORY = sys.argv[2] if len(sys.argv) > 2 else "/tmp/nyc_hls"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that ignores dead clients and stays quiet."""

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def log_message(self, fmt, *args):
        pass


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    handler = functools.partial(QuietHandler, directory=DIRECTORY)
    with Server(("127.0.0.1", PORT), handler) as httpd:
        print(f"nyc-cam bridge: serving {DIRECTORY} on http://127.0.0.1:{PORT}/live.m3u8", flush=True)
        httpd.serve_forever()
