from __future__ import annotations

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from motion_cog.library import DEFAULT_ARCHIVE, MotionLibrary


def make_handler(library: MotionLibrary):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def respond(self, status: int, body: bytes, content_type: str = 'application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def json_response(self, status: int, value: dict):
            self.respond(status, json.dumps(value).encode())

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, If-None-Match')
            self.end_headers()

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path == '/health':
                self.json_response(200, {'status': 'healthy', 'cached_clips': len(library.assets),
                                         'gpu_loaded': False, 'new_generation_enabled': False})
            elif url.path in {'/v1/animations/library', '/motions/kimodo/index.json'}:
                query = parse_qs(url.query).get('q', [''])[0][:300]
                self.json_response(200, library.index(query))
            elif url.path in library.files:
                body = library.files[url.path]
                etag = '"' + hashlib.sha256(body).hexdigest() + '"'
                if self.headers.get('If-None-Match') == etag:
                    self.send_response(304)
                    self.send_header('ETag', etag)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'public, max-age=31536000, immutable')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('ETag', etag)
                self.end_headers()
                self.wfile.write(body)
            else:
                self.json_response(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/v1/animations/generations':
                self.json_response(404, {'error': 'not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length < 1 or length > 64 << 10 or self.headers.get('Transfer-Encoding'):
                    self.json_response(413, {'error': 'bounded JSON body required'})
                    return
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError('request must be an object')
                result = library.reuse(request)
            except NotImplementedError as error:
                self.json_response(501, {'error': str(error)})
            except LookupError as error:
                self.json_response(404, {'error': str(error)})
            except (ValueError, TypeError, AttributeError) as error:
                self.json_response(400, {'error': str(error)})
            else:
                self.json_response(200, result)

        def log_message(self, _format, *_args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description='CPU-only reusable Kimodo motion library')
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9092)
    parser.add_argument('--export', type=Path)
    args = parser.parse_args()
    library = MotionLibrary(args.archive)
    if args.export:
        library.export(args.export)
        return
    server = HTTPServer((args.host, args.port), make_handler(library))
    print(f'Kimodo cached library: {len(library.assets)} clips; GPU generation disabled', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
