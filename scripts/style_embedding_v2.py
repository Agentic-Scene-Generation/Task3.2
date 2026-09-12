"""Strict, CPU-only style input adapter; no embedding/model dependencies."""
import gzip
import json
import re
from pathlib import Path

try:
    from .style_v2_contract import load_style_ontology, validate_style_annotation
except ImportError:
    from style_v2_contract import load_style_ontology, validate_style_annotation

DEFAULT_ONTOLOGY = Path(__file__).resolve().parents[1]/'configurations/style_ontology_bonn_v1.json'
HSSD_KEY_RE = re.compile(r'(?:[0-9a-f]{40}|xxxx[0-9a-fx]{36})')


def canonical_asset_uid(uid):
    bare=uid.removeprefix('hssd:').lower()
    if HSSD_KEY_RE.fullmatch(bare):return 'hssd:'+bare
    if re.fullmatch(r'(?:3dfuture|others):[^\s/\\]+',uid):return uid
    raise ValueError(f'unsupported asset UID: {uid}')


def load_confirmed_paths(path, ontology_path=DEFAULT_ONTOLOGY, *, all_sources=False):
    ontology = load_style_ontology(Path(ontology_path))
    opener = gzip.open if Path(path).suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        payload = json.load(f)
    records = payload.get('records', payload)
    result = {}
    for uid, record in records.items():
        if all_sources:
            aid=canonical_asset_uid(uid)
        else:
            aid = uid.removeprefix('hssd:').lower()
            if not HSSD_KEY_RE.fullmatch(aid):continue
        if not isinstance(record, dict) or record.get('style_status') not in ('source_metadata', 'visual_reviewed'):continue
        errors = validate_style_annotation(record, ontology)
        # Source-only children must carry the actual source dataset and label.
        for item in record.get('styles', []):
            if not isinstance(item, dict):continue
            child = ontology['level_2'].get(item.get('level_2_id'))
            if child and (item.get('source_dataset') not in child['source_datasets'] or
                          item.get('source_label') not in child['source_labels']):
                errors.append('unproven_source_child')
        if errors:
            raise ValueError(f'{uid}: invalid confirmed style: {", ".join(errors)}')
        paths = sorted([(x['level_1_id'], x.get('level_2_id'), x['evidence_type'])
                        for x in record['styles']], key=lambda x:(x[0],x[1] or ''))
        if aid in result and result[aid] != paths:
            raise ValueError(f'conflicting HSSD aliases: {uid}')
        result[aid] = paths
    return result
