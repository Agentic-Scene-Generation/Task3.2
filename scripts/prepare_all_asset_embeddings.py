"""Build an auditable all-source multimodal queue, including image-missing rows."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time
try:
    from .style_embedding_v2 import canonical_asset_uid, load_confirmed_paths
except ImportError:
    from style_embedding_v2 import canonical_asset_uid, load_confirmed_paths

def read(path):
    raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)

def style_text(paths):
    return ('Verified styles: '+'; '.join(a.replace('_',' ')+((' > '+b.replace('_',' ')) if b else '') for a,b,e in paths)+'.') if paths else ''

def content(uid, source, category, paths):
    return f'Asset UID: {uid}. Source: {source}. Category: {category}. '+style_text(paths)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--style-overlay',type=Path,required=True)
    p.add_argument('--evidence-root',type=Path,required=True)
    p.add_argument('--generated-root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    registry=read(a.data_root/'source_registry.json');config=read(a.data_root/'config.json')
    styles=load_confirmed_paths(a.style_overlay,all_sources=True)
    counts=Counter();ready=Counter();styled=Counter();pending=Counter();seen=set();sources=[]
    def emit(f,row):
        uid=row['asset_uid']
        if uid in seen:raise ValueError('duplicate UID '+uid)
        seen.add(uid);s=row['source'];counts[s]+=1
        ready[s]+=row['input_status']=='ready';styled[s]+=bool(row['style_paths'])
        if row['input_status']!='ready':pending[row['input_status']]+=1
        f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (a.output/'embedding_inputs.jsonl').open('w') as f:
        for catalog,spec in registry['sources'].items():
            path=a.data_root/spec['lookup'];raw=path.read_bytes()
            sources.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()})
            records=json.loads(gzip.decompress(raw))
            for key,r in records.items():
                uid=canonical_asset_uid(r.get('asset_uid',key));source=r.get('source',catalog)
                g=r.get('geometry_ref',{});root=config['geometry_roots'].get(g.get('root_key','hssd_materialized'))
                mesh=Path(root)/g['path'] if root and g.get('path') else None
                images=[];role=None;bundle=a.evidence_root/key.replace(':','_')
                manifest=bundle/'manifest.json'
                if manifest.exists():
                    evidence=read(manifest)
                    if canonical_asset_uid(evidence['asset_uid'])!=uid:raise ValueError('image UID mismatch')
                    images=[str(bundle/x['file']) for x in evidence.get('views',[])]
                    role='rendered_mesh_views'
                if not images and catalog=='3dfuture' and mesh:
                    preview=mesh.parent/'image.jpg'
                    if preview.exists():images=[str(preview)];role='dataset_product_preview'
                existing=bool(images) and all(Path(x).is_file() and Path(x).stat().st_size>0 for x in images)
                mesh_exists=mesh is not None and mesh.is_file()
                status='ready' if existing and mesh_exists else ('missing_geometry' if not mesh_exists else 'missing_images')
                paths=styles.get(uid,[])
                emit(f,{'asset_uid':uid,'source':source,'catalog':catalog,'category':r.get('category',''),
                     'asset_path':str(mesh) if mesh else None,'geometry_ref':g,
                     'image_paths':images,'image_role':role,'style_paths':paths,
                     'content':content(uid,source,r.get('category',''),paths),
                     'input_status':status,'membership':'registered_library'})
        if a.generated_root:
            index_path=a.generated_root/'ABO_ACTIVE_SCOPE_INDEX.json'
            index=read(index_path)
            sources.append({'path':str(index_path),'sha256':hashlib.sha256(index_path.read_bytes()).hexdigest()})
            for x in index['items']:
                aid=x['asset_id'];uid='generated:hunyuan:'+aid;folder=a.generated_root/aid
                # condition.png depicts the input photograph, NOT the generated mesh.
                # Never pretend it is a rendering of the generated asset.
                prov=read(folder/'provenance.json')['source'];category=prov.get('visual_screen',{}).get('result',{}).get('object_category','')
                emit(f,{'asset_uid':uid,'source':'generated_hunyuan','catalog':'generated_candidates',
                     'category':category,'asset_path':str(folder/'painted_pbr.glb'),
                     'image_paths':[],'image_role':None,'style_paths':[],
                     'content':content(uid,'generated_hunyuan',category,[]),
                     'input_status':'needs_generated_mesh_render','membership':'generated_candidate_not_canonical'})
    manifest={'time':time.time(),'source_counts':dict(counts),'ready_counts':dict(ready),
              'confirmed_style_counts':dict(styled),'pending_reasons':dict(pending),
              'total':len(seen),'ready':sum(ready.values()),'sources':sources,
              'style_overlay':str(a.style_overlay),'style_overlay_sha256':hashlib.sha256(a.style_overlay.read_bytes()).hexdigest(),
              'queue_sha256':hashlib.sha256((a.output/'embedding_inputs.jsonl').read_bytes()).hexdigest(),
              'vectors_computed':0,'scope':'all registered sources plus current-scope generated candidates; historical excluded generations not reintroduced'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='sources'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
