"""Prepare an isolated real-label embedding input; never modify canonical data."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

try:
    from .style_embedding_v2 import load_confirmed_paths, HSSD_KEY_RE
except ImportError:
    from style_embedding_v2 import load_confirmed_paths, HSSD_KEY_RE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotation-root',type=Path,required=True)
    p.add_argument('--reviews-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--all-sources',action='store_true',help='Include every registered source, preserving namespaced IDs')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('canonical_merge',args.annotation_root/'build/annotate_asset_styles_v2.py')
    merge=importlib.util.module_from_spec(spec);spec.loader.exec_module(merge)
    ontology_path=args.annotation_root/'data/style_ontology_bonn_v1.json'
    ontology=merge.load_style_ontology(ontology_path)
    source=args.annotation_root/'data/asset_style_annotations_v2.json.gz'
    records=json.loads(gzip.decompress(source.read_bytes()))
    files=[args.reviews_root/'results/adjudicated.jsonl',*sorted((args.reviews_root/'results_multikey').glob('shard_*/adjudicated.jsonl'))]
    reviews={};conflicts=set();provenance=[];rejected=Counter()
    for path in files:
        raw=path.read_bytes()
        provenance.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()})
        for line in raw.splitlines():
            if not line.strip():continue
            review=json.loads(line)
            uid=review['asset_uid']
            if not args.all_sources and not HSSD_KEY_RE.fullmatch(uid):continue
            if uid in reviews and reviews[uid]!=review:
                conflicts.add(uid)
            reviews[uid]=review
    valid={}
    for uid,review in reviews.items():
        if uid in conflicts:
            rejected['conflicting_duplicate']+=1;continue
        if review.get('style_status')!='visual_reviewed':continue
        labels=review.get('labels',[])
        if review.get('schema_version')!='style_visual_review@1.0' or review.get('run_count')!=2 or review.get('review_flags') or not labels:
            rejected['not_clean_two_run_review']+=1;continue
        if not all(re.fullmatch('[0-9a-f]{64}',str(review.get(k,''))) for k in ['evidence_sha256','prompt_sha256']):
            rejected['missing_evidence_hash']+=1;continue
        if not all(x.get('features') and all(f.get('view_id') and f.get('text') for f in x['features']) for x in labels):
            rejected['missing_view_features']+=1;continue
        valid[uid]=review
    merged=merge.merge_visual_reviews(records,valid,ontology)
    hssd=merged if args.all_sources else {k:v for k,v in merged.items() if HSSD_KEY_RE.fullmatch(k)}
    overlay=args.output/('all_confirmed_styles_v2.json.gz' if args.all_sources else 'hssd_confirmed_styles_v2.json.gz')
    overlay.write_bytes(gzip.compress(json.dumps({'records':hssd},ensure_ascii=False,sort_keys=True).encode(),mtime=0))
    paths=load_confirmed_paths(overlay,ontology_path,all_sources=args.all_sources)
    with (args.output/'style_text_inputs.jsonl').open('w') as f:
        for aid,styles in sorted(paths.items()):
            text='Verified styles: '+'; '.join(a.replace('_',' ')+((' > '+b.replace('_',' ')) if b else '') for a,b,_ in styles)+'.'
            f.write(json.dumps({'asset_id':aid,'style_text':text,'paths':styles},ensure_ascii=False)+'\n')
    report={'time':time.time(),'ontology_id':ontology['schema_version'],
            'ontology_sha256':hashlib.sha256(ontology_path.read_bytes()).hexdigest(),
            'source_overlay':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'review_inputs':provenance,'excluded_reviews':dict(rejected),'asset_records':len(hssd),
            'with_confirmed_style':len(paths),'without_confirmed_style':len(hssd)-len(paths),
            'level_1_counts':dict(Counter(a for styles in paths.values() for a,b,e in styles)),
            'overlay_sha256':hashlib.sha256(overlay.read_bytes()).hexdigest(),
            'scope':'isolated input snapshot; not canonical publication; no vectors recomputed; excludes round3 single-run candidates'}
    (args.output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
