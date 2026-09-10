"""Recover selected legacy object bytes without trusting ambiguous shard IDs.

This preserves checksum-matching bytes in a separate namespace. It does not assert
that their tokenization, document boundaries, provenance, or licensing are usable.
"""
import argparse,json,hashlib
from pathlib import Path
from .store import S3Store,canonical,digest


def recover(manifest_path,tokenizer_path,source_bucket,ids,destination,client):
    import ijson
    if not 1<=len(ids)<=128 or any(type(v)is not int or v<0 for v in ids):
        raise ValueError('Select 1–128 explicit legacy shard IDs')
    matches={value:[] for value in ids}
    with Path(manifest_path).open('rb') as f:
        for row in ijson.items(f,'shards.item'):
            if row.get('shard_id') in matches:matches[row['shard_id']].append(row)
    tokenizer=Path(tokenizer_path).read_bytes();tokenizer_root=destination.put(tokenizer)
    result={'schema':'neuroshard/legacy-recovery/v1','source_bucket':source_bucket,
        'tokenizer_sha256':tokenizer_root,'readiness':'quarantined: raw object integrity only; not activated for training','objects':[]}
    manifest_hash=hashlib.sha256()
    with Path(manifest_path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):manifest_hash.update(chunk)
    result['legacy_manifest_sha256']=manifest_hash.hexdigest()
    for value,rows in sorted(matches.items()):
        key=f'shard_{value}.pt'
        if not rows:
            result['objects'].append({'key':key,'status':'not_referenced'});continue
        response=client.get_object(Bucket=source_bucket,Key=key)
        try:raw=response['Body'].read(32*1024**2+1)
        finally:response['Body'].close()
        if len(raw)>32*1024**2:raise ValueError('Legacy object exceeds 32 MiB recovery limit')
        sha=digest(raw);valid=[row for row in rows if row['hash']==sha and row['size_bytes']==len(raw)]
        entry={'key':key,'actual_sha256':sha,'bytes':len(raw),'referenced_hashes':sorted({r['hash'] for r in rows}),
               'matching_rows':valid,'missing_hashes':sorted({r['hash'] for r in rows if r['hash']!=sha})}
        if valid:
            entry['recovered_sha256']=destination.put(raw);entry['status']='bytes_preserved'
        else:entry['status']='no_matching_manifest_entry'
        result['objects'].append(entry)
    return {'recovery_root':destination.put(canonical(result)),'manifest':result}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--bucket',required=True);parser.add_argument('--ids',type=int,nargs='+',required=True)
    parser.add_argument('--prefix',default='recovered/v1/sha256');parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();store=S3Store(args.bucket,args.prefix)
    result=recover(args.manifest,args.tokenizer,args.bucket,set(args.ids),store,store.client)
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'recovery_root':result['recovery_root'],'objects':len(result['manifest']['objects']),'report':str(args.report)}))


if __name__=='__main__':main()
