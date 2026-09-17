#!/usr/bin/env python3
"""Issue private, conditional upload capabilities for committed trial assets.

Invoke with the existing local data-bucket identity. Stdout is a private pipe
to the owner transfer process and must never be printed by the controller.
"""
import json
import os
from pathlib import Path
import re
import shlex
import sys

import boto3
from botocore.config import Config


def sign(request):
    objects = request['objects']
    if (set(request) != {'objects'} or not isinstance(objects, dict) or not 1 <= len(objects) <= 4096
            or sum(spec['bytes'] for spec in objects.values()) > 250*1024**3):
        raise ValueError('Bound content-addressed publication requests')
    for key, spec in objects.items():
        if (not re.fullmatch('[0-9a-f]{64}', key) or set(spec) != {'bytes'}
                or type(spec['bytes']) is not int or not 0 < spec['bytes'] <= 2*1024**3):
            raise ValueError('Require bounded immutable object identities')
    credentials = Path('/etc/neuroshard/data.env')
    if credentials.exists():
        for line in credentials.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            key, value = line.removeprefix('export ').split('=', 1)
            if key.startswith('AWS_'):
                os.environ[key] = ''.join(shlex.split(value, comments=True))
    client = boto3.client('s3', region_name='us-east-1', config=Config(signature_version='s3v4'))
    return {key: {'bytes': spec['bytes'], 'put': client.generate_presigned_url('put_object',
        Params={'Bucket': 'neuroshard-training-data', 'Key': 'research/native-expert-live-20260916/objects/'+key,
            'ServerSideEncryption': 'AES256', 'Metadata': {'sha256': key}, 'IfNoneMatch': '*'}, ExpiresIn=1800)}
        for key, spec in objects.items()}


if __name__ == '__main__':
    sys.stdout.write(json.dumps(sign(json.load(sys.stdin))))
