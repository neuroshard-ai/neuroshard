#!/usr/bin/env python3
"""Check a downloaded model and execution dataset against a published genesis."""
import os
os.environ.update(ATEN_CPU_CAPABILITY='default',MKL_ENABLE_INSTRUCTIONS='SSE4_2',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,json,platform
from pathlib import Path
from neuroshard.inference.profile import check
from neuroshard.demo.work import digest

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--genesis',type=Path,required=True)
parser.add_argument('--model-dir',type=Path,required=True)
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
genesis=json.loads(args.genesis.read_text());spec=genesis['app_state']['manifest']
actual=check(spec,args.model_dir,json.loads(args.data.read_text()))
result={'chain_id':genesis['chain_id'],'manifest_hash':digest(actual),'python':platform.python_version(),
        'platform':platform.machine(),'initial_validation_loss_hex':actual['initial_validation_loss_hex'],
        'numerical_conformance':actual['numerical_conformance'],'libraries':actual['libraries'],'passed':True}
raw=json.dumps(result,indent=2)+'\n'
if args.output:args.output.write_text(raw)
print(raw,end='')
