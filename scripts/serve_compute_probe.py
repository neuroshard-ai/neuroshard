#!/usr/bin/env python3
"""Private, fixed-task serving probe for the operated compute-group experiment.

Only committed development task IDs are accepted. This is a benchmark fixture,
not a public inference API, billing endpoint or permissionless provider service.
"""
import argparse
import hashlib
import json
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from neuroshard.evolution import cooperative as group
from neuroshard.evolution import reference as engine
from neuroshard.evolution import reference_data as data


class Answers:
    def __init__(self, records, generate, model_digest, capacity=256):
        self.records={r['id']:r for r in records}
        self.generate,self.model_digest,self.capacity=generate,model_digest,capacity
        self.lock=threading.Lock()
        self.cache=OrderedDict()

    def answer(self, request):
        if (not isinstance(request,dict) or set(request)!={'request_id','task_id'}
                or not isinstance(request['request_id'],str) or not 1<=len(request['request_id'])<=128
                or request['task_id'] not in self.records):
            raise ValueError('Unknown task or malformed request')
        identifier=request['request_id']
        with self.lock:
            if identifier in self.cache:
                old=self.cache[identifier]
                if old['task_id']!=request['task_id']:raise ValueError('Request identity reused for another task')
                self.cache.move_to_end(identifier)
                return {**old,'cached':True}
            generated=self.generate(self.records[request['task_id']])
            result={'request_id':identifier,'task_id':request['task_id'],
                    'model_digest':self.model_digest,'generation':generated,'cached':False}
            self.cache[identifier]=result
            if len(self.cache)>self.capacity:self.cache.popitem(last=False)
            return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--bind',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=9090)
    args=parser.parse_args()
    from transformers import AutoTokenizer
    runtime=engine.configure('cuda',2)
    prepared=json.loads((args.home/'prepared.json').read_bytes())
    plan=prepared['plan']
    records=data.read_records(args.home/'inputs/dev.jsonl',prepared['roles']['dev']['sha256'])
    receipt=json.loads((args.model_dir/'checkpoint.json').read_bytes())
    for name,digest in receipt['files'].items():
        if name=='optimizer.pt':continue
        if Path(name).name!=name or data.sha256(args.model_dir/name)!=digest:raise ValueError('Model checksum mismatch')
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False)
    if data.tokenizer_identity(tokenizer)!=prepared['tokenizer']:raise ValueError('Tokenizer mismatch')
    model=engine.load_model(args.model_dir,'cuda',plan['model']['parameters'])
    digest=group.parameter_digest(model)
    answers=Answers(records,lambda record:engine.generate(model,tokenizer,[record],'cuda',plan['generation_tokens'],1)[0],digest)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_json(self,status,value):
            payload=json.dumps(value).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            if self.path!='/health':self.send_json(404,{'error':'unknown route'});return
            self.send_json(200,{'ready':True,'model_digest':digest,'runtime':runtime})
        def do_POST(self):
            if self.path!='/infer':self.send_json(404,{'error':'unknown route'});return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=1024:raise ValueError('Invalid request size')
                self.connection.settimeout(60)
                request=json.loads(self.rfile.read(size))
                result=answers.answer(request)
                self.send_json(200,result)
            except (ValueError,TypeError):self.send_json(400,{'error':'invalid request'})
    print(json.dumps({'ready':True,'model_digest':digest,'started':time.time()}),flush=True)
    server=ThreadingHTTPServer((args.bind,args.port),Handler)
    server.daemon_threads=True
    server.serve_forever()


if __name__=='__main__':main()
