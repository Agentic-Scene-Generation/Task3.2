"""Index the namespaced all-source queue using the existing Qwen3-VL client."""
import argparse
import fcntl
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from itertools import islice
from pathlib import Path
import numpy as np
try:
    from . import index_rendered_hssd_assets_zvec as base
except ImportError:
    import index_rendered_hssd_assets_zvec as base

def document_id(uid):return hashlib.sha256(uid.encode()).hexdigest()

def prompt(row):
    return ('Represent this asset for semantic visual retrieval. '+row['content']+' '+
            ' '.join(f'view_{i}: <__media__>' for i in range(len(row['image_paths']))))

def embed_checked(client,row,images):
    hashes=[hashlib.sha256(x.read_bytes()).hexdigest() for x in images]
    vector=client.embed_images_text(images,prompt(row))
    if hashes!=[hashlib.sha256(x.read_bytes()).hexdigest() for x in images]:
        raise ValueError('images changed during request: '+row['asset_uid'])
    return vector,hashes

def commit_batch(collection,docs,records,ledger):
    """Only checkpoint flushed vectors; atomic ledger replacement tolerates crashes."""
    if not docs:return
    base.flush_docs(collection,docs)
    collection.flush()
    previous=ledger.read_bytes() if ledger.exists() else b''
    temporary=ledger.with_name(ledger.name+'.tmp')
    with temporary.open('wb') as f:
        f.write(previous)
        f.write(''.join(json.dumps(r)+'\n' for r in records).encode())
        f.flush();os.fsync(f.fileno())
    os.replace(temporary,ledger)

def verify_completed_input(item):
    row,expected=item
    if not Path(row['asset_path']).is_file():raise ValueError('completed asset missing: '+row['asset_uid'])
    actual=[hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in row['image_paths']]
    if actual!=expected:raise ValueError('completed input images changed; use new snapshot/collection')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--collection-path',type=Path,required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:8014')
    p.add_argument('--model-id',required=True,help='Exact serving model/checkpoint identity; must match on resume')
    p.add_argument('--dimension',type=int,default=2048)
    p.add_argument('--limit',type=int)
    p.add_argument('--worker-urls',nargs='+',help='Verified equivalent replicas; one request per replica, one index writer')
    p.add_argument('--backend-proof',type=Path,help='Required parity evidence for alternate execution backends')
    p.add_argument('--flush-size',type=int,default=256)
    a=p.parse_args()
    if a.flush_size<1:raise ValueError('flush-size must be positive')
    lockpath=a.collection_path.with_name(a.collection_path.name+'.lock')
    lockpath.parent.mkdir(parents=True,exist_ok=True)
    held=lockpath.open('a')
    fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if base.zvec is None:raise RuntimeError('zvec unavailable; no embeddings or index writes performed')
    client=base.LlamaEmbeddingClient(a.base_url,timeout_seconds=60,request_retries=0)
    client.props()  # Fail before opening any collection if endpoint is unavailable.
    clients=[client]
    if a.worker_urls:
        if not a.backend_proof:raise ValueError('alternate replicas require parity evidence')
        proof=json.loads(a.backend_proof.read_text())
        if proof.get('passed') is not True or proof.get('worker_urls')!=a.worker_urls:raise ValueError('unverified replicas')
        if a.model_id.endswith('_cpu') and proof.get('cpu_resume_compatible') is not True:
            raise ValueError('CPU/GPU parity failed; cannot mix vectors in CPU collection')
        clients=[base.LlamaEmbeddingClient(url,timeout_seconds=300,request_retries=0) for url in a.worker_urls]
        for worker in clients:worker.props()
    signature={'input_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
               'model_id':a.model_id,'dimension':a.dimension,'base_url':a.base_url}
    sidecar=a.collection_path.with_name(a.collection_path.name+'.input.json')
    if a.collection_path.exists() and not sidecar.exists():raise ValueError('unidentified existing collection')
    if sidecar.exists() and json.loads(sidecar.read_text())!=signature:raise ValueError('input/model changed; choose new collection')
    schema=base.make_schema('all_assets_style_v2',a.dimension,'flat',False)
    collection=base.open_or_create_collection(a.collection_path,schema,False)
    sidecar.write_text(json.dumps(signature,indent=2))
    ledger=a.collection_path.with_name(a.collection_path.name+'.done.jsonl')
    done={json.loads(line)['asset_uid']:json.loads(line)['images_sha256'] for line in ledger.read_text().splitlines()} if ledger.exists() else {}
    if done:
        rows=(json.loads(line) for line in a.manifest.read_text().splitlines())
        checks=((row,done[row['asset_uid']]) for row in rows if row['asset_uid'] in done)
        verified=0
        with ThreadPoolExecutor(max_workers=8) as audit_pool:
            while True:
                chunk=list(islice(checks,64))
                if not chunk:break
                list(audit_pool.map(verify_completed_input,chunk))
                verified+=len(chunk)
                if verified%1024==0:print(json.dumps({'resume_inputs_verified':verified,'total':len(done)}),flush=True)
        if verified!=len(done):raise ValueError('ledger contains assets outside manifest')
    def pending_rows():
        emitted=0
        with a.manifest.open() as inputs:
            for line in inputs:
                row=json.loads(line)
                if row['input_status']!='ready':continue
                if row['asset_uid'] in done:continue
                images=[Path(x) for x in row['image_paths']]
                if not Path(row['asset_path']).is_file() or not all(x.is_file() for x in images):raise ValueError('input missing: '+row['asset_uid'])
                yield row,images
                emitted+=1
                if a.limit and emitted>=a.limit:return
    pending=pending_rows()
    count=0;docs=[];records=[];started=time.monotonic()
    with ThreadPoolExecutor(max_workers=len(clients)) as pool:
        active={}
        def submit(worker):
            item=next(pending,None)
            if item is not None:
                row,images=item
                active[pool.submit(embed_checked,worker,row,images)]=(worker,row,images)
        for worker in clients:submit(worker)
        while active:
            completed,_=wait(active,return_when=FIRST_COMPLETED)
            for future in completed:
                worker,row,images=active.pop(future)
                vector,hashes=future.result()
                if len(vector)!=a.dimension or not np.isfinite(vector).all() or np.linalg.norm(vector)==0:raise ValueError('invalid embedding vector')
                fields={'asset_id':row['asset_uid'],'name':row['category'],'wordnet_key':'',
                        'object_groups':row.get('object_groups',[]),'views':[f'view_{i}' for i in range(len(images))],
                        'image_paths':row['image_paths'],'asset_path':row['asset_path'],'content':row['content']}
                docs.append(base.zvec.Doc(id=document_id(row['asset_uid']),vectors={'embedding':vector},fields=fields))
                records.append({'asset_uid':row['asset_uid'],'images_sha256':hashes})
                submit(worker)  # Refill this GPU without waiting for the other replicas.
                count+=1
                if len(docs)>=a.flush_size:
                    t=time.monotonic();commit_batch(collection,docs,records,ledger)
                    print(json.dumps({'new_vectors':count,'total_done':len(done)+count,'flush_seconds':time.monotonic()-t,'elapsed_seconds':time.monotonic()-started}),flush=True)
                    docs=[];records=[]
        commit_batch(collection,docs,records,ledger)
    print(json.dumps({'new_vectors':count,'already_done':len(done),'collection':str(a.collection_path)}))

if __name__=='__main__':main()
