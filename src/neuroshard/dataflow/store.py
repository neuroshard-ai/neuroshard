"""Content-addressed objects. A successful write never replaces different bytes."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

MAX_OBJECT_BYTES = 1024 ** 3


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA-256 digest")
    return value


class LocalStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, sha):
        return self.root / check_hash(sha)

    def put(self, data):
        if not isinstance(data, bytes) or len(data) > MAX_OBJECT_BYTES:
            raise ValueError("Object must be bytes, at most 1 GiB")
        sha = digest(data)
        fd, name = tempfile.mkstemp(prefix=".upload-", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data); f.flush(); os.fsync(f.fileno())
            try:
                os.link(name, self.path(sha))
            except FileExistsError:
                if self.get(sha) != data:
                    raise ValueError("Existing object does not match its content hash")
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.unlink(name)
        return sha

    def get(self, sha):
        with self.path(sha).open("rb") as f:
            data = f.read(MAX_OBJECT_BYTES + 1)
        if len(data) > MAX_OBJECT_BYTES or digest(data) != sha:
            raise ValueError("Stored object failed its SHA-256 check")
        return data


class S3Store:
    """Use the normal AWS credential chain; never embed credentials in manifests."""
    def __init__(self, bucket, prefix="datasets/v1/sha256", client=None):
        if not re.fullmatch(r"[a-zA-Z0-9/_-]+", prefix.strip("/")):
            raise ValueError("Invalid object prefix")
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client("s3", config=Config(connect_timeout=10, read_timeout=120,
                retries={"max_attempts": 3, "mode": "standard"}))
        self.client, self.bucket, self.prefix = client, bucket, prefix.strip("/")

    def key(self, sha):
        return f"{self.prefix}/{check_hash(sha)}"

    def put(self, data):
        if not isinstance(data, bytes) or len(data) > MAX_OBJECT_BYTES:
            raise ValueError("Object must be bytes, at most 1 GiB")
        sha = digest(data)
        from botocore.exceptions import ClientError
        try:
            self.client.put_object(Bucket=self.bucket, Key=self.key(sha), Body=data,
                IfNoneMatch="*", ChecksumSHA256=base64.b64encode(bytes.fromhex(sha)).decode(),
                ContentType="application/octet-stream")
        except ClientError as exc:
            if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 412:
                raise
            # An existing key is acceptable only after checking the actual bytes.
            if self.get(sha) != data:
                raise ValueError("Conflicting immutable S3 object") from exc
        return sha

    def get(self, sha):
        body = self.client.get_object(Bucket=self.bucket, Key=self.key(sha))["Body"]
        try:
            data = body.read(MAX_OBJECT_BYTES + 1)
        finally:
            body.close()
        if len(data) > MAX_OBJECT_BYTES or digest(data) != sha:
            raise ValueError("S3 object failed its SHA-256 check")
        return data
