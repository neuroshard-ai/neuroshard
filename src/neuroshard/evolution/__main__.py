"""Run explicit, bounded NeuroShard model-evolution experiments."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(prog='python -m neuroshard.evolution',description=__doc__,epilog='Experimental tools. These commands do not change the public testnet or issue spendable NEURO.')
    commands=parser.add_subparsers(dest='command',required=True)
    download=commands.add_parser('download-seed',help='Download and hash-check the pinned 135M starting model')
    download.add_argument('--model-dir',type=Path,required=True)
    convert=commands.add_parser('import-model',help='Convert a local tied-weight Llama seed into verifiable components')
    convert.add_argument('--model-dir',type=Path,required=True)
    convert.add_argument('--objects',type=Path,required=True)
    grow=commands.add_parser('grow',help='Create an identity-initialized capacity candidate; does not promote it')
    grow.add_argument('--objects',type=Path,required=True)
    grow.add_argument('--model-root',required=True)
    grow.add_argument('--layers',type=int,default=4)
    audit=commands.add_parser('audit',help='Independently replay one committed worker step')
    audit.add_argument('--objects',type=Path,required=True)
    audit.add_argument('--trace-root',required=True)
    inspect=commands.add_parser('inspect',help='Show model size and assignments under explicit worker capacities')
    inspect.add_argument('--objects',type=Path,required=True)
    inspect.add_argument('--model-root',required=True)
    inspect.add_argument('--workers',type=int,default=3)
    text=commands.add_parser('inspect-tokenizer',help='Verify the text codec used by a model and show its vocabulary')
    text.add_argument('--objects',type=Path,required=True)
    text.add_argument('--model-root',required=True)
    epoch=commands.add_parser('run-epoch',help='Collect, train and evaluate one durable research epoch within its daily budget')
    epoch.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='download-seed':
        from .seed import FILES,MODEL_REPO,MODEL_REVISION,verify
        from huggingface_hub import hf_hub_download
        for name in FILES:
            hf_hub_download(MODEL_REPO,name,revision=MODEL_REVISION,local_dir=args.model_dir)
        verify(args.model_dir)
        print(json.dumps({'model_dir':str(args.model_dir),'model':MODEL_REPO,'revision':MODEL_REVISION,'verified_files':len(FILES)}))
        return
    if args.command=='run-epoch':
        return run_epoch(args.config)
    from .objects import Objects
    store=Objects(args.objects)
    if args.command=='import-model':
        from .model import from_pretrained
        from .text import TextCodec,bind_model
        from transformers import AutoTokenizer
        root,model=from_pretrained(args.model_dir,store)
        tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False)
        codec=TextCodec(tokenizer,store)
        root,model=bind_model(store,root,codec)
        result={'model_root':root,'parameters':model['parameters'],'tokenizer_root':codec.root}
    elif args.command=='grow':
        from .model import grow
        root,model=grow(args.model_root,store,args.layers)
        result={'model_root':root,'parameters':model['parameters'],'status':'unevaluated_candidate'}
    elif args.command=='audit':
        from .worker import replay_trace
        result=replay_trace(store,args.trace_root)
    elif args.command=='inspect-tokenizer':
        from .text import TextCodec
        model=store.json(args.model_root)
        if 'tokenizer_root' not in model:
            raise ValueError('Legacy numerical model has no text contract; import the model with the current text profile')
        codec=TextCodec.load(store,model['tokenizer_root'])
        codec.check_model(model)
        result={'model_root':args.model_root,'tokenizer_root':codec.root,**codec.profile}
    else:
        from .model import place
        model=store.json(args.model_root)
        result={'model_root':args.model_root,'parameters':model['parameters'],
                'assignments':place(model,[48000000]*args.workers)}
    print(json.dumps(result,indent=2))


def run_epoch(path):
    from .objects import Objects
    from .seed import verify
    from .data import TextCorpus
    from .text import TextCodec,bind_model
    from .model import place
    from .pipeline import Pipeline
    from .transport import Endpoint
    from .controller import Epochs
    config=json.loads(path.read_bytes())
    base=path.resolve().parent
    def local(value):
        value=Path(value).expanduser()
        return value if value.is_absolute() else base/value
    model_dir=local(config['model_dir'])
    verify(model_dir)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    store=Objects(local(config['objects']))
    codec=TextCodec(tokenizer,store)
    initial_root,_=bind_model(store,config['initial_model_root'],codec)
    corpus=TextCorpus(local(config['corpus']),store,codec,**config.get('text',{}))
    train=corpus.register(config['training_source'])
    heldout=corpus.register(config['heldout_source'])
    capacities=[worker.get('capacity',48000000) for worker in config['workers']]
    endpoints=[Endpoint(worker['url'],local(worker['token_file']).read_text().strip(),store) for worker in config['workers']]
    def factory(root,name,journal):
        codec.check_model(store.json(root))
        partitions=place(store.json(root),capacities)
        return Pipeline(store,root,endpoints[:len(partitions)],capacities,name,journal=journal)
    epochs=Epochs(local(config['epochs']),store,corpus,factory,initial_root,train,heldout,
                  capacities=capacities,progress=lambda value:print(json.dumps(value),flush=True),**config.get('budget',{}))
    result=epochs.run_once()
    print(json.dumps({'status':result['status'],'tokenizer_root':codec.root,'accepted_research_model':epochs.accepted_root,
                      'candidate':result.get('candidate'),'decision':result.get('decision')},indent=2))


if __name__=='__main__':
    main()
