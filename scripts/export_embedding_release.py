"""Portable, checksum-verified float32 embedding release. No model calls."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate_records(rows):
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('duplicate ID')
    if '/data/250010098/' in json.dumps(rows):raise ValueError('private path')

def write_package(path,rows,vectors,signature):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    if (path/'manifest.json').exists():raise ValueError('release already exists')
    validate_records(rows);vectors=np.asarray(vectors,dtype=np.float32)
    if vectors.ndim!=2 or len(vectors)!=len(rows) or not np.isfinite(vectors).all():raise ValueError('invalid vectors')
    if not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=.001):raise ValueError('nonunit vectors')
    meta=path/'records.jsonl.gz'
    with gzip.open(meta,'wt',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    files=[{'name':meta.name,'sha256':sha(meta)}];shards=[]
    for start in range(0,len(rows),2048):
        p=path/f'vectors_{start//2048:03d}.npy';np.save(p,vectors[start:start+2048],allow_pickle=False)
        files.append({'name':p.name,'sha256':sha(p)});shards.append({'file':p.name,'start':start,'count':min(2048,len(rows)-start)})
    manifest={'schema':'portable_embedding@1','count':len(rows),'dimension':vectors.shape[1],'dtype':'float32','signature':signature,'files':files,'shards':shards}
    (path/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    verify_package(path)

def verify_package(path):
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    for f in m['files']:
        if Path(f['name']).name!=f['name'] or sha(path/f['name'])!=f['sha256']:raise ValueError('invalid file/checksum')
    with gzip.open(path/'records.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
    validate_records(rows)
    if len(rows)!=m['count']:raise ValueError('count mismatch')
    count=0
    for s in m['shards']:
        if Path(s['file']).name!=s['file']:raise ValueError('invalid shard path')
        a=np.load(path/s['file'],allow_pickle=False)
        if s['start']!=count or a.shape!=(s['count'],m['dimension']) or a.dtype!=np.float32 or not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a,axis=1),1,atol=.001):raise ValueError('invalid shard')
        count+=len(a)
    if count!=len(rows):raise ValueError('shard coverage')
    return m

def export(source,dest):
    import zvec
    source=Path(source)
    ids=[hashlib.sha256(json.loads(l)['asset_uid'].encode()).hexdigest() for l in source.with_name(source.name+'.done.jsonl').read_text().splitlines()]
    signature=json.loads(source.with_name(source.name+'.input.json').read_text());rows=[];vectors=[]
    c=zvec.open(str(source),option=zvec.CollectionOption(read_only=True))
    try:
        if len(ids)!=c.stats.doc_count:raise ValueError('ledger mismatch')
        for start in range(0,len(ids),256):
            docs=c.fetch(ids[start:start+256])
            for key in ids[start:start+256]:
                d=docs[key];rows.append({'id':key,'fields':d.fields});vectors.append(d.vectors['embedding'])
    finally:c.close()
    write_package(dest,rows,vectors,signature)

def restore(package,destination):
    import zvec
    m=verify_package(package);package=Path(package)
    if Path(destination).exists():raise ValueError('destination exists')
    with gzip.open(package/'records.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
    fields=[]
    for k,v in rows[0]['fields'].items():fields.append(zvec.FieldSchema(name=k,data_type=zvec.DataType.ARRAY_STRING if isinstance(v,list) else zvec.DataType.STRING,nullable=True))
    schema=zvec.CollectionSchema(name='all_assets_style_v2',fields=fields,vectors=[zvec.VectorSchema(name='embedding',data_type=zvec.DataType.VECTOR_FP32,dimension=m['dimension'],index_param=zvec.FlatIndexParam(metric_type=zvec.MetricType.COSINE))])
    c=zvec.create_and_open(str(destination),schema=schema)
    try:
        for s in m['shards']:
            a=np.load(package/s['file'],allow_pickle=False)
            for offset in range(0,len(a),128):
                docs=[zvec.Doc(id=r['id'],fields=r['fields'],vectors={'embedding':v.tolist()}) for r,v in zip(rows[s['start']+offset:s['start']+offset+128],a[offset:offset+128])]
                c.insert(docs)
        c.flush()
        if c.stats.doc_count!=m['count']:raise ValueError('restored count mismatch')
    finally:c.close()
    Path(str(destination)+'.input.json').write_text(json.dumps(m['signature'],indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['export','verify','restore']);p.add_argument('source',type=Path);p.add_argument('destination',type=Path,nargs='?');a=p.parse_args()
    if a.action=='verify':print(json.dumps(verify_package(a.source),indent=2))
    else:
        if a.destination is None:p.error('destination required')
        (export if a.action=='export' else restore)(a.source,a.destination)
