#!/usr/bin/env python3
"""Serve the VisionBridge site on the Tailscale interface.
"/" lands on the new v2 page; v1 stays reachable at /index.html."""
import http.server, os, socketserver

SITE = "/Users/scrimwiggins/.hermes/workspace/visionbridge-demo/site"
PORT = 8899
os.chdir(SITE)

class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", ""):
            self.path = "/index-v2.html"
        return super().do_GET()

    def log_message(self, fmt, *a):
        print("%s - %s" % (self.address_string(), fmt % a), flush=True)

class S(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

with S(("0.0.0.0", PORT), H) as httpd:
    print("serving %s on 0.0.0.0:%d" % (SITE, PORT), flush=True)
    httpd.serve_forever()
