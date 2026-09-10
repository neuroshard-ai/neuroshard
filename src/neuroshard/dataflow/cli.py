"""Prepare and verify bounded immutable datasets; ingestion never runs on import."""
import argparse,json
from pathlib import Path
from .ingest import Ingestor,verify_snapshot
from .store import LocalStore,S3Store


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objects',type=Path,help='Local content-addressed object directory')
    parser.add_argument('--bucket',help='S3 bucket; normal AWS credential chain')
    parser.add_argument('--prefix',default='datasets/v1/sha256')
    sub=parser.add_subparsers(dest='command',required=True)
    ingest=sub.add_parser('ingest',help='Resume a bounded JSONL text import')
    ingest.add_argument('input',type=Path);ingest.add_argument('--journal',type=Path,required=True)
    for key in ('origin','revision','license'):ingest.add_argument('--'+key,required=True)
    ingest.add_argument('--max-shards',type=int,default=16)
    ingest.add_argument('--batch-records',type=int,default=64)
    verify=sub.add_parser('verify',help='Check every object and document in a snapshot')
    verify.add_argument('sha256')
    args=parser.parse_args()
    if bool(args.objects)==bool(args.bucket):parser.error('Choose exactly one of --objects or --bucket')
    store=LocalStore(args.objects) if args.objects else S3Store(args.bucket,args.prefix)
    if args.command=='verify':result=verify_snapshot(store,args.sha256)
    else:
        ingestor=Ingestor(args.journal,store)
        try:
            value=ingestor.ingest(args.input,{key:getattr(args,key) for key in ('origin','revision','license')},
                                   args.batch_records,args.max_shards)
            result={'snapshot':value['sha256'],'shards':len(value['manifest']['shards']),
                    'next':'Run verify with this digest before activating the dataset.'}
        finally:ingestor.close()
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
