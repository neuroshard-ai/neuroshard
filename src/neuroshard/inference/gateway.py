"""Public LLM network reads and bounded signed-transaction relay."""
import argparse,base64,json,math,threading,time
from pathlib import Path
from urllib.parse import urlparse,parse_qs

from neuroshard.demo import client,protocol,work
from neuroshard.publicnet import gateway as base


class Gateway(base.Gateway):
    def model(self):
        summary=self.summary();spec=self.cached('manifest',lambda:client.query(self.rpc,'/manifest'),60)
        model=spec['model'];config=json.loads((Path(self.config['model_dir'])/'config.json').read_text())
        return {'chain_id':summary['chain_id'],'height':summary['height'],'round':summary['round'],
            'model_root':summary['model_root'],'serving_root':summary['serving_root'],'genesis_sha256':self.config['genesis_sha256'],
            'parameter_count':134515008+2*model['adapter_rank']*model['hidden_size'],
            'base_parameter_count':134515008,'trainable_parameter_count':2*model['adapter_rank']*model['hidden_size'],
            'model_name':'SmolLM2-135M-Instruct + NeuroShard adapter','model_source':model['repo'],'model_revision':model['revision'],
            'execution':{'model':{'num_layers':config['num_hidden_layers'],'hidden_dim':config['hidden_size'],
                'num_heads':config['num_attention_heads'],'num_kv_heads':config['num_key_value_heads'],
                'intermediate_dim':config['intermediate_size'],'vocab_size':config['vocab_size'],'max_seq_len':model['max_input_tokens']},
                'optimizer':'SGD on residual adapter; frozen pretrained backbone','learning_rate':model['learning_rate'],
                'batch_size':1,'sequence_length':model['training_sequence_tokens'],'numerics':model['arithmetic'],
                'dataset_sha256':spec['dataset_sha256']},
            'last_training_loss':float.fromhex(summary['last_training_loss_hex']) if summary['last_training_loss_hex'] else None,
            'validation_loss':float.fromhex(summary['validation_loss_hex']),
            'serving_validation_loss':float.fromhex(summary['serving_loss_hex']),
            'last_training_evaluation':summary['last_training_evaluation'],
            'metric':'Training minibatch and four fixed held-out sequences; not a general-purpose quality benchmark'}

    def inference_info(self):
        summary=self.summary();params=summary['params']
        config_path=self.home/'services.json'
        services=json.loads(config_path.read_text()) if config_path.exists() else {}
        heartbeat_path=Path(services.get('provider_status_file',self.home/'provider-status.json'))
        try:
            heartbeat=json.loads(heartbeat_path.read_text())
            provider_online=(heartbeat['public_key']==services.get('provider') and heartbeat['chain_id']==self.config['chain_id']
                             and 0<=time.time()-heartbeat['checked_at']<240)
        except (OSError,ValueError,KeyError,TypeError):provider_online=False
        return {'chain_id':self.config['chain_id'],'genesis_sha256':self.config['genesis_sha256'],
            'model_root':summary['serving_root'],'model_name':'SmolLM2-135M-Instruct + NeuroShard adapter',
            'provider':services.get('provider'),'provider_online':provider_online,'fee_atoms':str(params['fee']),
            'price_per_max_token_atoms':str(params['inference_token_price']),'max_tokens':64,
            'request_lifetime_blocks':params['inference_blocks'],'pending_requests':summary['pending_inference'],
            'public_prompts':True,'verification':'Every validator replays inference before native settlement',
            'scope':'Experimental CPU service; prompts and responses are public; fixed price for the requested token limit'}


def validate_rpc(value):
    if (isinstance(value,dict) and value.get('method')=='abci_query' and isinstance(value.get('params'),dict)
            and value['params'].get('path') in ('/jobs','/job')):
        # Reuse all envelope/parameter checks, extending only the bounded query names.
        checked={**value,'params':{**value['params'],'path':'/task'}}
        base.validate_rpc(checked)
        return value['method'],value['params']
    return base.validate_rpc(value)


def handler(gateway):
    parent=base.handler(gateway)
    class Handler(parent):
        def do_GET(self):
            parsed=urlparse(self.path);path=parsed.path;query=parse_qs(parsed.query)
            if path not in ('/api/inference','/api/inference/request','/network/dataset.json','/api/account'):
                return super().do_GET()
            try:
                if gateway.limited(base.client_ip(self.client_address[0],self.headers.get('X-Real-IP'))):
                    return self.respond(429,{'error':'Request budget exceeded'})
                if path=='/api/inference':return self.respond(200,gateway.inference_info())
                if path=='/network/dataset.json':return self.respond(200,(gateway.home/'dataset.json').read_bytes())
                if path=='/api/account':
                    value=client.query(gateway.rpc,'/account',{'public_key':query['public_key'][0]})
                    for key in ('balance','locked','total'):value[key]=str(value[key])
                    return self.respond(200,value)
                value=client.query(gateway.rpc,'/job',{'id':query['id'][0]})
                value.pop('weights',None)
                for key in ('price','provider_paid','refunded'):
                    if key in value:value[key]=str(value[key])
                return self.respond(200,value)
            except (ValueError,KeyError,TypeError,OverflowError) as exc:return self.respond(400,{'error':str(exc)})
            except OSError:return self.respond(503,{'error':'Native full node unavailable'})

        def do_POST(self):
            try:
                if self.path!='/rpc':return self.respond(404,{'error':'Unknown endpoint'})
                if gateway.limited(base.client_ip(self.client_address[0],self.headers.get('X-Real-IP'))):
                    return self.respond(429,{'error':'Request budget exceeded'})
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=32768 or self.headers.get('Transfer-Encoding'):raise ValueError('Invalid RPC request size')
                raw=self.rfile.read(length)
                if len(raw)!=length:raise ValueError('Truncated request')
                request=protocol.parse_json(raw);method,params=validate_rpc(request)
                result=client.rpc(gateway.rpc,method,params,timeout=90)
                return self.respond(200,{'jsonrpc':'2.0','id':request.get('id'),'result':result})
            except (ValueError,KeyError,TypeError,OverflowError,RecursionError) as exc:
                return self.respond(400,{'error':str(exc)})
            except OSError:return self.respond(503,{'error':'Native full node unavailable; query your request id before retrying'})
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--home',type=Path,required=True);args=parser.parse_args()
    gateway=Gateway(args.home);server=base.BoundedServer(('127.0.0.1',gateway.config['base_port']+3),handler(gateway))
    stop=threading.Event();follower=threading.Thread(target=gateway.history.follow,args=(gateway,stop),daemon=True);follower.start()
    try:server.serve_forever()
    finally:stop.set();server.server_close();follower.join(timeout=5)


if __name__=='__main__':main()
