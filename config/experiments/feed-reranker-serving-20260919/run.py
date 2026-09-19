"""Execute the frozen full-service diagnostic from its separate source home."""
import json
from pathlib import Path

import ordinary_cloud
from run_lossless_diagnostic import run

HOME = Path('/home/ubuntu/neuroshard/.neuroshard/feed-reranker-serving-20260919')

if __name__ == '__main__':
    deployment = json.loads((HOME/'deployment.json').read_bytes())
    ordinary_cloud.REPO = deployment['source_home']
    ordinary_cloud.PYTHON = ordinary_cloud.REPO+'/.neuroshard/venv/bin/python'
    run(HOME, HOME)
