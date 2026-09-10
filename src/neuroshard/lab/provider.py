"""Separate full-model inference provider for paid candidate jobs."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from neuroshard.demo import protocol, work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    work.configure_cpu()
    identity = protocol.Identity.load_or_create(args.key)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= work.MAX_MESSAGE_BYTES or self.path != "/infer":
                    raise ValueError("Invalid inference request")
                body = protocol.parse_json(self.rfile.read(size))
                output = work.infer(body["weights"], **body["request"])
                result = {"output": output, "receipt": identity.sign({"task_id": body["task_id"],
                    "model_root": output["model_root"], "request_root": work.digest(body["request"]),
                    "result_root": work.digest(output)})}
                code = 200
            except (ValueError, KeyError, TypeError) as exc:
                result, code = {"error": str(exc)}, 400
            raw = work.canonical(result)
            self.send_response(code)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
