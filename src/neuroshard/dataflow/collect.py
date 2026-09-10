"""Collect a finite batch from a pinned upstream revision and publish an immutable snapshot."""
import argparse,fcntl,hashlib,json,os,re
from pathlib import Path
from .store import S3Store,LocalStore,canonical,digest
from .ingest import Ingestor


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def collect(config,home,store,rows=None,render=None):
    home=Path(home);home.mkdir(parents=True,exist_ok=True)
    if not re.fullmatch('[0-9a-f]{40}',config['revision']):raise ValueError('Pin the upstream dataset to a full commit hash')
    if not 1<=config['batch_documents']<=1024 or not 1<=config['max_documents']<=100000:
        raise ValueError('Set explicit batch and collection bounds')
    identity={k:config[k] for k in ('repo','revision','split','license','tokenizer_revision')}
    identity_root=digest(canonical(identity))
    with (home/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        progress=json.loads((home/'progress.json').read_text()) if (home/'progress.json').exists() else {'identity':identity_root,'cursor':0}
        if progress['identity']!=identity_root:raise ValueError('A new source revision requires a separate collection home')
        pending_path=home/'pending.json';input_path=home/'pending.jsonl'
        if pending_path.exists():
            pending=json.loads(pending_path.read_text())
            if progress['cursor']==pending['end']:
                pending_path.unlink();input_path.unlink(missing_ok=True)
            elif progress['cursor']!=pending['start']:raise ValueError('Collection journal cursor conflict')
        if progress['cursor']>=config['max_documents']:
            return {**progress,'status':'collection_budget_complete'}
        if not pending_path.exists():
            count=min(config['batch_documents'],config['max_documents']-progress['cursor'])
            if rows is None:
                from datasets import load_dataset
                from pyarrow.dataset import ParquetFragmentScanOptions
                rows=iter(load_dataset(config['repo'],revision=config['revision'],split=config['split'],streaming=True,
                    fragment_scan_options=ParquetFragmentScanOptions(pre_buffer=False)).skip(progress['cursor']).take(count))
            if render is None:
                from transformers import AutoTokenizer
                model=Path(config['model_dir']);tokenizer_file=model/'tokenizer.json'
                if digest(tokenizer_file.read_bytes())!=config['tokenizer_sha256']:raise ValueError('Collector tokenizer checksum mismatch')
                if digest((model/'tokenizer_config.json').read_bytes())!=config['tokenizer_config_sha256']:raise ValueError('Collector chat template checksum mismatch')
                tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
                render=lambda item:tokenizer.apply_chat_template(item['messages'],tokenize=False,add_generation_prompt=False)
            records=[]
            try:
                for _,item in zip(range(count),rows):records.append({'text':render(item)})
            finally:
                # A bounded consumer stops before EOF. Release Arrow's streaming
                # fragments while Python is still alive, not during interpreter shutdown.
                close=getattr(rows,'close',None)
                if close:close()
                rows=None;close=None
                import gc
                gc.collect()
            if not records:return {**progress,'status':'upstream_exhausted'}
            raw=b''.join(canonical(item)+b'\n' for item in records)
            with input_path.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            pending={'start':progress['cursor'],'end':progress['cursor']+len(records),'sha256':digest(raw)}
            save(pending_path,pending)
        pending=json.loads(pending_path.read_text())
        if digest(input_path.read_bytes())!=pending['sha256']:raise ValueError('Pending collection input checksum mismatch')
        ingest=Ingestor(home/'ingest.sqlite',store)
        try:
            value=ingest.ingest(input_path,{'origin':f'https://huggingface.co/datasets/{config["repo"]}; {config["split"]} rows {pending["start"]}:{pending["end"]}; chat template {config["tokenizer_revision"]}',
                'revision':config['revision'],'license':config['license']},batch_records=64,max_shards=1024)
        finally:ingest.close()
        progress={**progress,'cursor':pending['end'],'snapshot':value['sha256'],'status':'snapshot_published',
                  'activation':'Requires an explicit execution-profile update; existing chains keep their pinned dataset'}
        save(home/'progress.json',progress)
        pending_path.unlink();input_path.unlink()
        return progress


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--home',type=Path,required=True);args=parser.parse_args();config=json.loads(args.config.read_text())
    store=S3Store(config['bucket'],config.get('prefix','datasets/v1/sha256')) if 'bucket' in config else LocalStore(config['objects'])
    print(json.dumps(collect(config,args.home,store),indent=2))


if __name__=='__main__':main()
