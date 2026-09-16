"""Bounded immutable object transfer with full hash readback before acceptance.

URLs come from the operator's configured object store, never from work claims.
An ambiguous PUT may have succeeded: a retry uses the same bytes and conditional
creation, then verifies the complete GET. Errors deliberately omit signed URLs.
"""
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile
import time

import requests

BLOCK = 4 * 1024**2
TRANSIENT = {408, 409, 425, 429, 500, 502, 503, 504}
# The 1.7B tied embedding with FP32 Adam state is about 1.21 GB.
MAX_OBJECT_BYTES = 2 * 1024**3


class CorruptObject(ValueError):
    pass


class UnavailableObject(RuntimeError):
    pass


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            value.update(block)
    return value.hexdigest()


def transfer(key, size, get_url, *, source=None, put_url=None, destination=None,
             attempts=5, max_seconds=600):
    """Upload/read back or atomically restore one content-addressed object."""
    if (not isinstance(key, str) or not re.fullmatch('[0-9a-f]{64}', key)
            or type(size) is not int or not 0 < size <= MAX_OBJECT_BYTES
            or type(attempts) is not int or not 1 <= attempts <= 8
            or type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
            or not 0 < max_seconds <= 1800 or (source is None) != (put_url is None)):
        raise ValueError('Require a bounded committed object transfer')
    if source is not None:
        source = Path(source)
        if source.is_symlink() or source.stat().st_size != size or digest(source) != key:
            raise CorruptObject('Local object differs from its commitment')
    if destination is not None:
        destination = Path(destination)
        if destination.is_symlink():
            raise ValueError('Object destination may not be a symlink')
        if destination.exists():
            if destination.stat().st_size != size or digest(destination) != key:
                raise CorruptObject('Existing destination differs from its commitment')
            return {'bytes': size, 'readback_verified': True, 'attempts': 0}
        destination.parent.mkdir(parents=True, exist_ok=True)
    end = time.monotonic() + max_seconds

    def remaining():
        seconds = end - time.monotonic()
        if seconds <= 0:
            raise TimeoutError('Object transfer deadline expired')
        return (min(15, seconds), min(180, seconds))

    for attempt in range(attempts):
        pending = None
        try:
            if source is not None:
                # Reopen at byte zero after every ambiguous network failure.
                with source.open('rb') as stream, requests.put(put_url, data=stream,
                        headers={'x-amz-server-side-encryption': 'AES256',
                                 'x-amz-meta-sha256': key, 'If-None-Match': '*'},
                        timeout=remaining()) as response:
                    if response.status_code in TRANSIENT:
                        raise requests.ConnectionError('Transient object-store response')
                    if response.status_code not in (200, 201, 412):
                        raise UnavailableObject('Immutable PUT refused with HTTP ' + str(response.status_code))
            with requests.get(get_url, stream=True, timeout=remaining()) as response:
                if response.status_code in TRANSIENT:
                    raise requests.ConnectionError('Transient object-store response')
                if response.status_code != 200:
                    raise UnavailableObject('Object GET refused with HTTP ' + str(response.status_code))
                output = None
                if destination is not None:
                    output = tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.receiving-', delete=False)
                    pending = Path(output.name)
                try:
                    received, value = 0, hashlib.sha256()
                    for block in response.iter_content(BLOCK):
                        remaining()
                        received += len(block)
                        if received > size:
                            raise CorruptObject('Object exceeds its committed length')
                        value.update(block)
                        if output is not None:
                            output.write(block)
                    if received != size or value.hexdigest() != key:
                        raise CorruptObject('Object readback differs from its commitment')
                    if output is not None:
                        output.flush()
                        os.fsync(output.fileno())
                finally:
                    if output is not None:
                        output.close()
            remaining()
            if pending is not None:
                os.replace(pending, destination)
                pending = None
                descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            return {'bytes': size, 'readback_verified': True, 'attempts': attempt + 1}
        except (requests.RequestException, TimeoutError) as error:
            if attempt + 1 == attempts or time.monotonic() >= end:
                raise UnavailableObject('Object ' + key + ' unavailable: ' + type(error).__name__) from None
            time.sleep(min(.25 * 2**attempt, max(0, end - time.monotonic())))
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)
    raise UnavailableObject('Object transfer exhausted its bound')


def restore(key, size, urls, destination, *, attempts=6, max_seconds=600):
    """Restore the same committed bytes across configured replicas and retries.

    Corrupt or unavailable responses never become a local accepted object. A
    different replica may still provide the exact bytes. URL and authentication
    strings stay out of returned diagnostics, including after total failure.
    """
    if (not isinstance(urls, (list, tuple)) or not 1 <= len(urls) <= 8
            or any(not isinstance(url, str) or not url.startswith(('https://', 'http://')) for url in urls)
            or type(attempts) is not int or not 1 <= attempts <= 8
            or type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
            or not 0 < max_seconds <= 1800):
        raise ValueError('Require bounded configured object replicas')
    # A corrupt preexisting cache is a local fault; do not repeatedly fetch
    # replicas only to have transfer reject the same existing destination.
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        return {**transfer(key, size, urls[0], destination=destination,
                           attempts=1, max_seconds=max_seconds), 'replica': None, 'failures': []}
    deadline, failures = time.monotonic() + max_seconds, []
    for attempt in range(attempts):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        replica = attempt % len(urls)
        try:
            result = transfer(key, size, urls[replica], destination=destination,
                              attempts=1, max_seconds=remaining)
            return {**result, 'attempts': attempt + 1, 'replica': replica, 'failures': failures}
        except (CorruptObject, UnavailableObject) as error:
            failures.append({'replica': replica, 'error': type(error).__name__})
            if attempt + 1 < attempts:
                time.sleep(min(.5 * 2**attempt, max(0, deadline - time.monotonic())))
    raise UnavailableObject('Object ' + key + ' unavailable after ' + str(len(failures)) + ' replica attempts')
