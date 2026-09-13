#!/usr/bin/env python3
"""Prepare, train and score the operated one/two-GPU learning comparison.

This driver executes trusted research jobs. It does not join a permissionless
compute market, change native consensus, issue tokens or promote serving.
"""
import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import timedelta
from pathlib import Path

from neuroshard.evolution import cooperative as group
from neuroshard.evolution import grounded_tasks as tasks
from neuroshard.evolution import reference as engine
from neuroshard.evolution import reference_data as data
from neuroshard.dataflow.store import canonical


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("clean-single", "damaged-single", "clean-pair")


def implementation():
    paths = ["scripts/run_cooperative_learning.py", "src/neuroshard/evolution/cooperative.py",
             "src/neuroshard/evolution/grounded_tasks.py", "src/neuroshard/evolution/reference.py",
             "src/neuroshard/evolution/reference_data.py", "src/neuroshard/dataflow/store.py",
             "docs/learning-reference-requirements.txt"]
    return data.identity({name: data.sha256(ROOT / name) for name in paths})


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def model_snapshot(directory, expected):
    receipt = json.loads((directory / "snapshot.json").read_bytes())
    if receipt["model"] != expected:
        raise ValueError("Wrong upstream seed")
    if not {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"} <= receipt["files"].keys():
        raise ValueError("Incomplete model snapshot")
    for name, digest in receipt["files"].items():
        if Path(name).name != name or data.sha256(directory / name) != digest:
            raise ValueError("Seed snapshot checksum mismatch")
    return receipt


def validate_plan(plan):
    if plan["format"] != "neuroshard-cooperative-learning-v1" or plan["purpose"] != "development":
        raise ValueError("Use an explicit development plan")
    if tuple(plan["arms"]) != ARMS or plan["cooperation"]["world_size"] != 2:
        raise ValueError("This experiment requires its three declared arms")
    for name in ("grounded_train_documents", "replay_documents", "dev_documents", "test_documents"):
        data.integer(plan[name], 4, 4096, name)
    train = plan["training"]
    for name, low, high in [("steps",1,2048),("batch_documents",2,128),("checkpoint_steps",1,2048)]:
        data.integer(train[name], low, high, name)
    if train["batch_documents"] % 2 or train["checkpoint_steps"] > train["steps"]:
        raise ValueError("Use even batches and an attainable checkpoint interval")
    for name, limit in [("learning_rate", .001), ("weight_decay", 1), ("clip_norm",10)]:
        if type(train[name]) not in (int,float) or not 0 < train[name] <= limit:
            raise ValueError("Invalid optimizer setting")
    data.integer(train["warmup_steps"],0,train["steps"]-1,"warmup")
    data.integer(plan["generation_tokens"],1,256,"generation limit")
    data.integer(plan["max_length"],32,2048,"context limit")
    data.integer(plan["grounded_token_weight"],1,16,"grounded target weight")
    data.integer(plan["budget"]["seconds"],1,28800,"time budget")
    data.integer(plan["budget"]["disk_gib"],1,160,"disk budget")
    return plan


def tokenizer_for(directory):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)


def verify_seed(directory, prepared):
    if model_snapshot(directory,prepared['plan']['model'])!=prepared['model_snapshot']:
        raise ValueError('Seed files differ from the committed preparation')


def binding_for(prepared, runtime, arm):
    return data.identity({'prepared':data.identity(prepared),'profile':group.runtime_profile(runtime),
                          'arm':arm,'world':2 if arm=='clean-pair' else 1})


def prepare(args, plan):
    home = args.home
    if any(home.glob("*")):
        raise ValueError("Prepare requires an empty experiment directory")
    reference = args.reference_home
    if data.sha256(reference / "prepared.json") != plan["reference_prepared_sha256"]:
        raise ValueError("Unexpected source reference preparation")
    old = json.loads((reference / "prepared.json").read_bytes())
    snapshot = model_snapshot(args.model_dir, plan["model"])
    tokenizer = tokenizer_for(args.model_dir)
    if data.tokenizer_identity(tokenizer) != old["tokenizer"]:
        raise ValueError("Tokenizer differs from the source reference")
    replay = data.read_records(reference / "inputs/train.jsonl", old["roles"]["train"]["sha256"])
    replay = replay[:plan["replay_documents"]]
    if len(replay) != plan["replay_documents"]:
        raise ValueError("Insufficient public replay records")
    retention = data.read_records(reference / "inputs/retention.jsonl", old["roles"]["retention"]["sha256"])
    inputs, roles, prompts = home / "inputs", {}, set()
    inputs.mkdir(parents=True, exist_ok=True)
    clean_train = None
    for role in ("test", "dev", "train"):
        count = plan["grounded_train_documents"] if role == "train" else plan[f"{role}_documents"]
        records = []
        for index in range(count):
            case = tasks.make_case(plan["task_seed"], role, index)
            prompt = tasks.prompt(case)
            if prompt in prompts:
                raise ValueError("Duplicate exact prompt across roles")
            prompts.add(prompt)
            answer = json.dumps(tasks.expected(case), separators=(",", ":"))
            if not tasks.check_answer(case, answer)["correct"]:
                raise ValueError("Generated target fails its executable check")
            messages = [{"role":"user", "content":prompt}, {"role":"assistant", "content":answer}]
            records.append({"id":tasks.task_identity(case), "task":case, "messages":messages,
                            "loss_weight":plan["grounded_token_weight"],
                            **data.conversation(tokenizer,messages,plan["max_length"])})
        if role == "train":
            clean_train = records
            records = records + replay
        roles[role] = write_records(inputs / f"{role}.jsonl", records)
    damaged, changed = [], []
    for record in clean_train:
        if int(record["id"],16) % 4 == 0:
            answer = tasks.damaged_target(record["task"])
            if tasks.check_answer(record["task"],answer)["correct"]:
                raise ValueError("Control corruption unexpectedly remains correct")
            messages = [record["messages"][0], {"role":"assistant", "content":answer}]
            record = {**record,"messages":messages,**data.conversation(tokenizer,messages,plan["max_length"])}
            changed.append(record["id"])
        damaged.append(record)
    roles["damaged"] = write_records(inputs / "damaged.jsonl",damaged + replay)
    roles["retention"] = write_records(inputs / "retention.jsonl",retention)
    prepared = {"plan":plan,"implementation":implementation(),"model_snapshot":snapshot,
                "tokenizer":data.tokenizer_identity(tokenizer),"roles":roles,"damaged_ids":changed,
                "retention_scope":"Previously exposed public response-loss probes; no fresh broad-capability claim"}
    data.save(home / "prepared.json",prepared)
    emit("prepared",sha256=data.sha256(home / "prepared.json"),damaged_targets=len(changed))


def write_records(path, records):
    with path.open("xb") as output:
        for record in records:
            output.write(canonical(record)+b"\n")
        output.flush(); os.fsync(output.fileno())
    return {"sha256":data.sha256(path),"ids":[r["id"] for r in records],
            "documents":len(records),"targets":sum(r["targets"] for r in records),
            "weighted_targets":sum(r["targets"]*r.get("loss_weight",1) for r in records)}


def prepared_inputs(args, plan):
    prepared = json.loads((args.home / "prepared.json").read_bytes())
    if prepared["plan"] != plan or prepared["implementation"] != implementation():
        raise ValueError("Plan or source changed after preparation")
    return prepared


def committed_preparation(prepared):
    path=ROOT / 'config/experiments/cooperative-learning-data-selection.json'
    relative=path.relative_to(ROOT).as_posix()
    committed=subprocess.check_output(['git','show',f'HEAD:{relative}'],cwd=ROOT)
    if committed!=path.read_bytes() or json.loads(committed)!=prepared:
        raise ValueError('Commit the exact prepared inputs before training or evaluation')


def partition(home, prepared, role, final_test=False):
    if role == "test" and not final_test:
        raise ValueError("Training and development cannot read final test inputs")
    records = data.read_records(home / f"inputs/{role}.jsonl",prepared["roles"][role]["sha256"])
    if [r["id"] for r in records] != prepared["roles"][role]["ids"]:
        raise ValueError("Partition identities changed")
    return records


def train(args, plan):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    rank, world = int(os.environ.get("RANK","0")), int(os.environ.get("WORLD_SIZE","1"))
    expected_world = 2 if args.arm == "clean-pair" else 1
    if world != expected_world or not 0 <= rank < world:
        raise ValueError("Arm and process-group size disagree")
    prepared = prepared_inputs(args,plan)
    committed_preparation(prepared)
    runtime = engine.configure(args.device,args.threads)
    binding = binding_for(prepared,runtime,args.arm)
    out = args.home / args.arm
    out.mkdir(exist_ok=True)
    with (out / f"rank-{rank}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = out / f"rank-{rank}-run.json"
        if state_path.exists():
            state = json.loads(state_path.read_bytes())
            if state["binding"] != binding:
                raise ValueError("Resume numerical identity changed")
            if state.get("completed"):
                emit("already_completed",arm=args.arm,rank=rank); return
        else:
            state={"binding":binding,"runtime":runtime,"started":time.time(),"rank":rank,"world":world}
            data.save(state_path,state)
        budget=engine.Budget(out,state["started"],plan["budget"])
        budget.check(force_disk=True)
        if world>1:
            dist.init_process_group("nccl" if args.device=="cuda" else "gloo",timeout=timedelta(seconds=900))
            group.agree_digest(binding,world,args.device)
        try:
            pointer = json.loads((out / "latest.json").read_bytes()) if (out / "latest.json").exists() else None
            receipt=None
            if pointer:
                directory,receipt=engine.verify_checkpoint(out,pointer,binding)
            else:
                if any(out.glob("checkpoint-*")):
                    raise ValueError("Checkpoint without pointer: explicitly recover its verified receipt first")
                directory=args.model_dir
                verify_seed(directory,prepared)
            tokenizer=tokenizer_for(args.model_dir)
            if data.tokenizer_identity(tokenizer)!=prepared["tokenizer"]:
                raise ValueError("Tokenizer identity changed")
            torch.manual_seed(plan["training"]["seed"])
            model=engine.load_model(directory,args.device,plan["model"]["parameters"])
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            model.train()
            optimizer=engine.optimizer_for(model,plan["training"])
            if pointer:
                engine.restore_optimizer(directory,optimizer,args.device)
            group.agree_digest(group.parameter_digest(model),world,args.device)
            wrapped=DistributedDataParallel(model,device_ids=[0] if args.device=="cuda" else None,
                broadcast_buffers=False,gradient_as_bucket_view=True,bucket_cap_mb=64) if world>1 else model
            train_records=partition(args.home,prepared,"damaged" if args.arm=="damaged-single" else "train")
            recipe=plan["training"]
            schedule=engine.schedule(len(train_records),recipe["steps"],recipe["batch_documents"],recipe["seed"])
            journal=receipt["records"] if receipt else []
            if any(r["step"]!=i+1 or r["documents"]!=[train_records[j]["id"] for j in schedule[i]] for i,r in enumerate(journal)):
                raise ValueError("Checkpoint journal differs from fixed global schedule")
            started=time.monotonic()
            network_start=network_bytes()
            for index in range(len(journal),recipe["steps"]):
                rows=[train_records[i] for i in schedule[index]]
                observation=group.step(wrapped,optimizer,rows,args.device,recipe,index,rank,world,budget.check)
                journal.append({k:v for k,v in observation.items() if k!="local_documents"})
                emit("step",arm=args.arm,rank=rank,**observation)
                if (index+1)%recipe["checkpoint_steps"]==0 or index+1==recipe["steps"]:
                    if rank==0:
                        pointer=engine.checkpoint(out,model,tokenizer,optimizer,index+1,binding,journal)
                        budget.check(force_disk=True)
                        emit("checkpoint",**pointer)
                    if world>1:
                        dist.barrier()
            digest=group.parameter_digest(model)
            group.agree_digest(digest,world,args.device)
            result={"arm":args.arm,"rank":rank,"world":world,"binding":binding,"parameter_digest":digest,
                    "candidate":pointer if rank==0 else None,"runtime":runtime,"steps":journal,
                    "seconds":time.monotonic()-started,"network_start":network_start,"network_end":network_bytes(),
                    "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated() if args.device=="cuda" else 0,
                    "tokens_issued":0,"serving_approved":False}
            data.save(out / f"rank-{rank}-result.json",result)
            state["completed"]=True;data.save(state_path,state)
            emit("completed",arm=args.arm,rank=rank,parameter_digest=digest,seconds=result["seconds"])
        finally:
            if world>1 and dist.is_initialized():dist.destroy_process_group()


def network_bytes():
    rows={}
    for line in Path('/proc/net/dev').read_text().splitlines()[2:]:
        name,values=line.split(':');values=values.split()
        if name.strip()!='lo':rows[name.strip()]={"rx":int(values[0]),"tx":int(values[8])}
    return rows


def selection(args, plan):
    prepared=prepared_inputs(args,plan)
    candidates={}
    for arm in ARMS:
        result=json.loads((args.home / arm / 'rank-0-result.json').read_bytes())
        if result['binding']!=binding_for(prepared,result['runtime'],arm):
            raise ValueError('Candidate belongs to a different preparation or arm')
        directory,receipt=engine.verify_checkpoint(args.home / arm,result['candidate'],result['binding'])
        if receipt['step']!=plan['training']['steps']:
            raise ValueError('Select only the predetermined final step')
        candidates[arm]={"candidate":result['candidate'],"binding":result['binding'],
                         "parameter_digest":result['parameter_digest']}
    data.save(args.home / 'selection.json',{"prepared":data.identity(prepared),"candidates":candidates})
    emit('selection_written',file=str(args.home / 'selection.json'))


def committed_selection(path, prepared, arm, candidate):
    relative=path.resolve().relative_to(ROOT)
    tracked=subprocess.check_output(['git','show',f'HEAD:{relative.as_posix()}'],cwd=ROOT)
    if tracked!=path.read_bytes():raise ValueError('Candidate selection is not committed at HEAD')
    selection=json.loads(tracked)
    if selection['prepared']!=data.identity(prepared):raise ValueError('Selected preparation differs')
    if arm!='seed' and selection['candidates'].get(arm)!=candidate:
        raise ValueError('The exact candidate must be committed before final-test scoring')


def evaluate(args, plan):
    import torch
    prepared=prepared_inputs(args,plan)
    committed_preparation(prepared)
    runtime=engine.configure(args.device,args.threads)
    candidate=None
    if args.arm=='seed':
        verify_seed(args.model_dir,prepared);directory=args.model_dir
    else:
        result=json.loads((args.home / args.arm / 'rank-0-result.json').read_bytes())
        if group.runtime_profile(runtime)!=group.runtime_profile(result['runtime']):
            raise ValueError('Evaluation numerical profile differs from training')
        if result['binding']!=binding_for(prepared,runtime,args.arm):
            raise ValueError('Candidate belongs to a different preparation or arm')
        directory,_=engine.verify_checkpoint(args.home / args.arm,result['candidate'],result['binding'])
        candidate={key:result[key] for key in ['candidate','binding','parameter_digest']}
    if args.role=='test':
        if args.selection is None:raise ValueError('Final test requires committed selection')
        committed_selection(args.selection,prepared,args.arm,candidate)
    output=args.home / 'evaluation' / f'{args.arm}-{args.role}.json'
    if output.exists():raise ValueError('Preserve completed evaluation; choose a fresh run')
    output.parent.mkdir(exist_ok=True)
    marker=output.with_suffix('.started.json')
    if marker.exists():raise ValueError('Interrupted evaluation requires explicit inspection; do not silently rescore')
    data.save(marker,{"prepared":data.identity(prepared),"candidate":candidate,"started":time.time(),"runtime":runtime})
    budget=engine.Budget(args.home / 'evaluation',time.time(),plan['budget'])
    model=engine.load_model(directory,args.device,plan['model']['parameters'])
    tokenizer=tokenizer_for(args.model_dir)
    records=partition(args.home,prepared,args.role,final_test=args.role=='test')
    generations=engine.generate(model,tokenizer,records,args.device,plan['generation_tokens'],len(records),budget.check)
    checks=[{"id":r['id'],"family":r['task']['family'],"variant":r['task']['variant'],
             **tasks.check_answer(r['task'],g['text'])} for r,g in zip(records,generations)]
    retention=engine.score(model,partition(args.home,prepared,'retention'),args.device,budget.check)
    report={"arm":args.arm,"role":args.role,"prepared":data.identity(prepared),"candidate":candidate,
            "runtime":runtime,"generations":generations,"checks":checks,"retention":retention,
            "correct":sum(c['correct'] for c in checks),"documents":len(checks),
            "seconds":time.time()-json.loads(marker.read_bytes())['started'],"serving_approved":False}
    data.save(output,report);emit('evaluated',arm=args.arm,role=args.role,correct=report['correct'],documents=len(checks))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','train','select','evaluate'])
    parser.add_argument('--home',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=ROOT / 'config/experiments/cooperative-learning.json')
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--reference-home',type=Path)
    parser.add_argument('--arm',choices=['seed',*ARMS],default='clean-single')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--role',choices=['dev','test'],default='dev')
    parser.add_argument('--selection',type=Path)
    args=parser.parse_args();args.home=args.home.resolve();args.model_dir=args.model_dir.resolve()
    plan=validate_plan(json.loads(args.plan.read_bytes()))
    if args.command=='train' and args.arm=='seed':raise ValueError('Seed is evaluation-only')
    {'prepare':prepare,'train':train,'select':selection,'evaluate':evaluate}[args.command](args,plan)


if __name__=='__main__':main()
