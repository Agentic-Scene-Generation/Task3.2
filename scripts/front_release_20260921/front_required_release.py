"""Publish total usable fronts; never confuse a fallback with certified semantics."""
import collections, copy, gzip, hashlib, json, math, shutil, sys, time
from pathlib import Path

# Host-bound publisher. Pure resolve() and its tests are portable; publication
# deliberately requires the existing authorized project and its release tooling.
BASE=Path('/data/250010098/codex_communication/3_scenesmith_asset_relations/feedback')
SHARED=Path('/data/task3_2/share_data/scenesmith/all_assets_style_v2_delivery_20260909')
LOCAL=SHARED/'asset_library/data'
REPO=BASE/'repos/hssd-publish-20260914'
TASK=BASE/'repos/task32-publish-20260914'
ROOTS=[LOCAL,REPO/'data',TASK/'scenesmith/scenebenchmark_critic/asset_annotation_data']
FILES=['hssd_annotation_lookup.json.gz','3dfuture/3dfuture_annotation_lookup.json.gz','external/others_annotation_lookup.json.gz','generated_annotations.json.gz']
RELEASE='front_required_20260921'
OUT=SHARED/RELEASE
MANIFEST='FRONT_REQUIRED_20260921.json'

def load(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    temp=p.with_name(p.name+'.front-release-tmp')
    temp.write_bytes(gzip.compress(raw,mtime=0) if p.suffix=='.gz' else raw)
    temp.chmod(0o666);temp.replace(p)

def horizontal(v):
    if not isinstance(v,(list,tuple)) or len(v)!=3 or any(type(x) not in (float,int) or not math.isfinite(x) for x in v):return None
    norm=math.sqrt(sum(x*x for x in v))
    if norm<1e-8 or abs(v[1])/norm>0.15:return None
    # Preserve exact normalized horizontal vector; no unreported cardinal snapping.
    norm=math.hypot(v[0],v[2])
    return [v[0]/norm,0.,v[2]/norm]

def resolve(row,candidate=None):
    old=row.get('canonical_front') or {}
    c=candidate or {};result=c.get('result') or {};status=result.get('front_status','not_judged')
    previous=horizontal(old.get('canonical_orientation_axis')) or horizontal(old.get('asset_local_front_axis'))
    axis=previous or [0.,0.,1.]
    source='previous_direction_as_fallback' if previous else 'canonical_positive_z_fallback'
    confidence=.2;semantic=False
    if not c and previous and old.get('canonical_orientation_is_semantic_front'):
        source='previous_semantic_pending_luna';semantic=True
        confidence=old.get('canonical_orientation_confidence',old.get('confidence',.2))
    reliable=min(result.get('confidence',0),result.get('up_confidence',0))>=.7 and not c.get('consistency_flags')
    chosen=None
    if reliable and status=='unique':chosen=horizontal(c.get('front_axis_evidence_local'))
    if reliable and status=='multiple_valid':
        chosen=next((v for a in result.get('alternative_front_axes',[]) if (v:=horizontal(a))),None)
    if chosen:
        axis=chosen;source='luna_v2_unique' if status=='unique' else 'luna_v2_one_of_multiple_valid'
        confidence=result['confidence'];semantic=True
    return {
        'schema_version':'canonical_front_required@1.0',
        'asset_local_front_axis':axis,'canonical_orientation_axis':axis,
        'canonical_orientation_axis_frame':'normalized_asset_local_y_up',
        'canonical_orientation_is_semantic_front':semantic,
        'canonical_orientation_source':source,'canonical_orientation_confidence':confidence,
        'canonical_front_direction':f'{axis} normalized asset-local Y-up',
        'canonical_front_face':'Luna visual semantic face' if source.startswith('luna_') else 'retained or conventional placement face',
        'up_axis':'y','up_axis_local':[0.,1.,0.],
        'confidence':confidence,'front_axis_confidence':confidence,
        'method':source,'status':'semantic_single_pass' if semantic else 'explicit_nonsemantic_fallback',
        'validation_status':'usable_axis_checked_not_independent_semantic_certification',
        'is_strict_front':False,'is_strict_positive_front':False,
        'luna_front_status':status,'luna_consistency_flags':c.get('consistency_flags',[]),
        'luna_up_confidence':result.get('up_confidence'),
        'semantic_direction_kind':'semantic_front' if semantic else 'fallback_front',
        'semantic_directions':[{'kind':'semantic_front' if semantic else 'fallback_front','axis':axis,'axis_frame':'normalized_asset_local_y_up','confidence':confidence,'is_primary':True,'is_strict_positive_front':False}],
        'canonical_orientation_policy':'Every published asset has a usable horizontal direction. Original Luna status is retained; fallback is a placement convention, not evidence of semantic front. Mesh upright is not changed.',
        'world_front_axis':None,'front_view_image_index':None,'front_view_image_name':None,
        'publication_release':RELEASE,
    }

def uid(key,row,filename):
    if row.get('asset_uid'):return row['asset_uid']
    return ('generated:'+key) if filename.startswith('generated') else (key if ':' in key else 'hssd:'+key)

def main():
    assert not OUT.exists(),'immutable_release_already_exists'
    OUT.mkdir();OUT.chmod(0o777)
    fronts=load(LOCAL/'front_visual_candidates_v2.json.gz')
    # Existing published judgments stay immutable. Add newly completed pending records only.
    sys.path.insert(0,str(BASE))
    import publish_annotations_20260921 as old_export
    live=load(old_export.FRONT/'manifest.json')['items']
    for item in live:
        if item['asset_uid'] not in fronts:
            u,r,e=old_export.front_record(item)
            if r:fronts[u]=r
    genitems=load(old_export.GEN/'INDEX.json')['items']
    generated=load(LOCAL/FILES[-1])
    for item in genitems:
        if item['asset_id'] not in generated:
            aid,row=old_export.generated(item,fronts);generated[aid]=row
    # Rebase the inherited scale provenance of new generated rows.
    raw=json.dumps(generated,ensure_ascii=False).replace('/data/250010098/abo_production_20260906/scale_reference.json',str(LOCAL/'annotation_provenance/scale_reference.json'))
    generated=json.loads(raw)
    backup=Path('/mnt/aoss2/codex_backup')/RELEASE
    baselines={};counts=collections.Counter();methods=collections.Counter();allfronts={}
    for index,root in enumerate(ROOTS):
        for name in FILES:
            p=root/name;baselines[str(p)]=sha(p)
            dest=backup/str(index)/name;dest.parent.mkdir(parents=True,exist_ok=True)
            assert not dest.exists(),'backup_exists';shutil.copy2(p,dest)
    for name in FILES:
        local=generated if name==FILES[-1] else load(LOCAL/name)
        for key,row in local.items():
            u=uid(key,row,name);candidate=fronts.get(u,row.get('front_visual_candidate'))
            if candidate:row['front_visual_candidate']=candidate
            if 'canonical_front_previous' not in row:row['canonical_front_previous']=copy.deepcopy(row.get('canonical_front'))
            row['canonical_front']=resolve(row,candidate)
            allfronts[u]=row['canonical_front'];counts[u.split(':')[0]]+=1
            methods[row['canonical_front']['canonical_orientation_source']]+=1
            if name==FILES[-1]:
                row['missing_families']=[x for x in row.get('missing_families',[]) if x!='canonical_front']
                # A usable front does not claim completion of other annotation families.
                row['annotation_complete']=False
        write(OUT/'shared'/name,local)
        portable=copy.deepcopy(local) if name==FILES[-1] else load(REPO/'data'/name)
        for key,row in portable.items():
            for field in ['canonical_front','canonical_front_previous','front_visual_candidate']:
                if field in local[key]:row[field]=local[key][field]
        write(OUT/'portable'/name,portable)
        del local,portable
    write(OUT/'canonical_front_required.json.gz',allfronts)
    manifest={'release':RELEASE,'snapshot_time':time.time(),'counts':dict(counts),'total':len(allfronts),'directions':dict(methods),
        'all_assets_have_nonempty_front':True,'fallback_is_not_semantic':True,'mesh_upright_changed':False,
        'luna_judgments_in_snapshot':len(fronts),'generated_index_count':len(genitems),'front_quality_certified':False,
        'embedding_vectors_updated':False,'scope':'Published inventory at snapshot; new producer outputs require subsequent publication. No Flare Luna requests.',
        'backup':str(backup),'documentation':'docs/FRONT_REQUIRED_20260921_ZH.md'}
    for kind in ['shared','portable']:
        manifest[kind+'_sha256']={n:sha(OUT/kind/n) for n in FILES}
    manifest['front_overlay_sha256']=sha(OUT/'canonical_front_required.json.gz')
    write(OUT/MANIFEST,manifest)
    # Validate all staged records before touching active consumer files.
    for kind in ['shared','portable']:
        for name in FILES:
            data=load(OUT/kind/name)
            for key,row in data.items():
                cf=row['canonical_front'];axis=cf['canonical_orientation_axis']
                assert horizontal(axis)==axis and abs(sum(x*x for x in axis)-1)<1e-6
                assert cf['asset_local_front_axis']==axis and isinstance(cf['canonical_front_direction'],str)
                assert cf==allfronts[uid(key,row,name)]
            assert b'/data/250010098/' not in gzip.decompress((OUT/kind/name).read_bytes()),name
    for index,root in enumerate(ROOTS):
        kind='shared' if index==0 else 'portable'
        for name in FILES:
            assert sha(root/name)==baselines[str(root/name)],'consumer_changed_during_export'
            target=root/name;temp=target.with_name(target.name+'.front-release-tmp')
            shutil.copyfile(OUT/kind/name,temp);temp.chmod(0o666);temp.replace(target)
        shutil.copyfile(OUT/'canonical_front_required.json.gz',root/'canonical_front_required.json.gz')
        m={**manifest,'sha256':manifest[kind+'_sha256']};write(root/MANIFEST,m)
        write(root/'CURRENT_ANNOTATION_RELEASE.json',{'release':RELEASE,'manifest':MANIFEST,'documentation':manifest['documentation'],'front_candidates_are_not_canonical_replacements':False,'embedding_vectors_updated':False})
    for p in OUT.rglob('*'):p.chmod(0o777 if p.is_dir() else 0o666)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
