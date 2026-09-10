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


def parquet_rows(paths,start,count):
    """Read only the requested rows, without asynchronous Arrow file callbacks."""
    import pyarrow.parquet as parquet
    for path in paths:
        with parquet.ParquetFile(path) as reader:
            if start>=reader.metadata.num_rows:
                start-=reader.metadata.num_rows
                continue
            for batch in reader.iter_batches(batch_size=64,columns=['messages'],use_threads=False):
                for item in batch.to_pylist():
                    if start:start-=1;continue
                    yield item
                    count-=1
                    if count==0:return


def upstream_rows(config,home,start,count):
    from huggingface_hub import HfApi,hf_hub_url
    import requests
    if not re.fullmatch('[A-Za-z0-9_-]+',config['split']):raise ValueError('Invalid split name')
    info=HfApi().repo_info(config['repo'],repo_type='dataset',revision=config['revision'],files_metadata=True)
    if info.sha!=config['revision']:raise ValueError('Upstream revision mismatch')
    pattern=re.compile(r'data/'+re.escape(config['split'])+r'-[0-9]+-of-[0-9]+\.parquet')
    files=sorted((f for f in info.siblings if pattern.fullmatch(f.rfilename)),key=lambda f:f.rfilename)
    if not files:raise ValueError('Pinned source must contain data/SPLIT-N-of-N.parquet files')
    cache=Path(home)/'upstream';cache.mkdir(exist_ok=True)
    def paths():
        for item in files:
            expected=item.lfs.sha256 if item.lfs else None
            if not expected or not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('Parquet source lacks a SHA-256 commitment')
            if not item.size or item.size>512*1024**2:raise ValueError('Upstream Parquet file exceeds 512 MiB limit')
            path=cache/(expected+'.parquet')
            if not path.exists():
                temporary=path.with_suffix('.part');size=0;hasher=hashlib.sha256()
                try:
                    url=hf_hub_url(config['repo'],item.rfilename,repo_type='dataset',revision=config['revision'])
                    with requests.get(url,stream=True,timeout=(15,120)) as response,temporary.open('wb') as out:
                        response.raise_for_status()
                        if not response.url.startswith('https://'):raise ValueError('Insecure upstream redirect')
                        for chunk in response.iter_content(1024**2):
                            size+=len(chunk)
                            if size>item.size:raise ValueError('Upstream file exceeds declared size')
                            hasher.update(chunk);out.write(chunk)
                        out.flush();os.fsync(out.fileno())
                    if size!=item.size or hasher.hexdigest()!=expected:raise ValueError('Upstream Parquet checksum mismatch')
                    os.replace(temporary,path)
                finally:temporary.unlink(missing_ok=True)
            hasher=hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda:source.read(1024**2),b''):hasher.update(chunk)
            if path.stat().st_size!=item.size or hasher.hexdigest()!=expected:raise ValueError('Cached Parquet checksum mismatch')
            yield path
    yield from parquet_rows(paths(),start,count)


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
                rows=upstream_rows(config,home,progress['cursor'],count)
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
                # Closing the generator also closes an early-exit Parquet reader.
                close=getattr(rows,'close',None)
                if close:close()
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
