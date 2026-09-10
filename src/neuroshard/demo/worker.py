"""A process that holds and trains one actual NeuroLLM pipeline stage."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from neuroshard.demo import protocol, work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=(0, 1), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--key", type=Path, required=True)
    args = parser.parse_args()
    work.configure_cpu()
    identity = protocol.Identity.load_or_create(args.key)
    stage = work.Stage(args.stage)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            raw = work.canonical(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != "/identity":
                return self.respond(404, {"error": "Unknown path"})
            self.respond(200, {"public_key": identity.public_key, "address": identity.address,
                               "stage": args.stage,
                               "parameter_count": sum(p.numel() for p in stage.parameters())})

        def do_POST(self):
            try:
                if self.path != "/compute":
                    return self.respond(404, {"error": "Unknown path"})
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= work.MAX_MESSAGE_BYTES:
                    raise ValueError("Invalid request length")
                request = protocol.parse_json(self.rfile.read(length))
                if request.get("operation") not in ("forward", "backward"):
                    raise ValueError("Unknown operation")
                if args.stage == 1 and request["operation"] != "backward":
                    raise ValueError("Stage one returns loss and gradients together")
                result = stage.compute(request)
                if "receipt" in result:
                    result["receipt"] = identity.sign(result["receipt"])
                self.respond(200, result)
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                self.respond(400, {"error": str(exc)})

    # Serialized requests also serialize PyTorch RNG/parameter mutation.
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Stage {args.stage}: {identity.address}, port {args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
