"""Validate token batches and optional assistant-only target masks."""


def unpack(batch, vocabulary):
    labels=None
    if isinstance(batch,dict):
        if set(batch)!={'input_ids','labels'}:
            raise ValueError('Unexpected labeled-batch fields')
        ids,labels=batch['input_ids'],batch['labels']
    else:
        ids=batch
    if (not isinstance(ids,list) or not 1<=len(ids)<=4 or
            any(not isinstance(row,list) or not 2<=len(row)<=256 for row in ids) or
            len({len(row) for row in ids})!=1 or sum(len(row) for row in ids)>512 or
            any(type(i) is not int or not 0<=i<vocabulary for row in ids for i in row)):
        raise ValueError('Batch outside execution bounds')
    if labels is not None:
        if (not isinstance(labels,list) or len(labels)!=len(ids) or any(
                not isinstance(row,list) or len(row)!=len(tokens) or
                any(type(label) is not int or label not in (-100,token) for label,token in zip(row,tokens)) or
                not any(label!=-100 for label in row[1:])
                for row,tokens in zip(labels,ids))):
            raise ValueError('Targets must be input tokens or the explicit ignore marker')
    return ids,labels


def from_windows(store,roots,tokenizer_root=None):
    windows=[store.json(key) for key in roots]
    identities={window.get('tokenizer_root') for window in windows}
    if len(identities)>1 or (tokenizer_root is not None and identities!={tokenizer_root}):
        raise ValueError('Token windows belong to different tokenizer contracts')
    if any(identity is not None for identity in identities) and any('labels' not in window for window in windows):
        raise ValueError('Text-contract windows require explicit response targets')
    ids=[window['tokens'] for window in windows]
    if any('labels' in window for window in windows):
        return {'input_ids':ids,'labels':[window.get('labels',window['tokens']) for window in windows]}
    return ids


def response_window(messages,tokenizer,context=64,response=64):
    """Keep actual response targets; pad only after them, with ignored labels."""
    assistant=next((i for i,m in enumerate(messages) if i>0 and m['role']=='assistant'),None)
    if assistant is None:
        return None
    prefix=tokenizer.apply_chat_template(messages[:assistant],tokenize=True,add_generation_prompt=True)
    complete=tokenizer.apply_chat_template(messages[:assistant+1],tokenize=True,add_generation_prompt=False)
    if complete[:len(prefix)]!=prefix:
        raise ValueError('Chat template does not have a stable assistant prefix')
    prompt=prefix[-context:]
    answer=complete[len(prefix):][:response]
    if len(prompt)<2 or len(answer)<4:
        return None
    padding=context+response-len(prompt)-len(answer)
    return {'tokens':prompt+answer+[tokenizer.eos_token_id]*padding,
            'labels':[-100]*len(prompt)+answer+[-100]*padding,
            'context_tokens':len(prompt),'response_tokens':len(answer)}
