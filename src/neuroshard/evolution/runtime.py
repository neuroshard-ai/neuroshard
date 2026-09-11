"""Declared numerical runtime; cross-machine conformance is still required."""
import platform
import sys
from importlib.metadata import version

VERSIONS = {'torch':'2.9.1+cpu','numpy':'2.2.6','safetensors':'0.7.0',
            'cryptography':'50.0.1','grpcio':'1.83.1','protobuf':'6.33.6'}


def check():
    if platform.system()!='Linux' or platform.machine()!='x86_64' or sys.byteorder!='little':
        raise RuntimeError('The experimental numerical profile requires Linux x86_64, little endian')
    if sys.version_info[:2] not in ((3,10),(3,11),(3,12)):
        raise RuntimeError('Use Python 3.10–3.12 for this execution profile')
    for name,expected in VERSIONS.items():
        if version(name)!=expected:
            raise RuntimeError(f'Execution profile requires {name}=={expected}')
    return {'packages':VERSIONS,'dtype':'float32','device':'cpu','threads':1,
            'aten_cpu_capability':'DEFAULT','mkl_instructions':'SSE4_2','mkldnn':False}
