"""Content-addressed local artifacts; hashes identify tensors and model revisions."""
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from neuroshard.dataflow.store import canonical

MAX_OBJECT_BYTES = 256 * 1024 * 1024


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class Objects:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fetchers = []

    def path(self, key):
        if not isinstance(key, str) or not re.fullmatch('[0-9a-f]{64}', key):
            raise ValueError('Invalid artifact digest')
        return self.root / key[:2] / key

    def put(self, raw):
        if len(raw) > MAX_OBJECT_BYTES:
            raise ValueError('Artifact exceeds object size bound')
        key = digest(raw)
        path = self.path(key)
        try:
            path.parent.mkdir()
        except FileExistsError:
            pass
        else:
            directory = os.open(self.root,os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(directory)
            finally: os.close(directory)
        if path.exists():
            if digest(path.read_bytes()) != key:
                raise ValueError('Corrupted existing artifact')
            return key
        fd, temporary = tempfile.mkstemp(prefix='.write-', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as target:
                target.write(raw)
                target.flush()
                os.fsync(target.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if digest(path.read_bytes()) != key:
                    raise ValueError('Concurrent artifact differs')
            directory = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(directory)
            finally: os.close(directory)
        finally:
            os.unlink(temporary)
        return key

    def get(self, key):
        path = self.path(key)
        if not path.exists():
            for fetch in self.fetchers:
                raw = fetch(key)
                if raw is not None:
                    if digest(raw) != key:
                        raise ValueError('Remote artifact checksum mismatch')
                    self.put(raw)
                    break
        if path.stat().st_size > MAX_OBJECT_BYTES:
            raise ValueError('Oversized artifact')
        raw = path.read_bytes()
        if digest(raw) != key:
            raise ValueError('Artifact checksum mismatch')
        return raw

    def put_json(self, value):
        return self.put(canonical(value))

    def json(self, key):
        return json.loads(self.get(key))

    def tensors(self, key, shapes=None):
        from safetensors.torch import load
        raw = self.get(key)
        if shapes is not None:
            header_size = int.from_bytes(raw[:8], 'little')
            if not 2 <= header_size <= 65536:
                raise ValueError('Invalid tensor header size')
            header = json.loads(raw[8:8+header_size])
            if set(header) != set(shapes) or any(
                header[name]['shape'] != shape or header[name]['dtype'] != 'F32'
                for name, shape in shapes.items()
            ):
                raise ValueError('Tensor shape or dtype differs from model commitment')
        return load(raw)

    def put_tensors(self, tensors):
        from safetensors.torch import save
        clean = {name: value.detach().cpu().contiguous() for name, value in sorted(tensors.items())}
        return self.put(save(clean))
