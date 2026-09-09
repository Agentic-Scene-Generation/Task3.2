import copy
import json
from pathlib import Path
import pytest
from scripts.style_embedding_v2 import load_confirmed_paths

AID='a'*40

def record():
    return {'schema_version':'asset_style@2.0','style_status':'source_metadata',
            'provenance':{'ontology_id':'bonn_furniture_styles_hierarchical@1.0'},
            'styles':[{'level_1_id':'traditional','level_2_id':'european_classic',
                       'evidence_type':'source_metadata','confidence':.9,
                       'source_dataset':'3dfuture','source_label':'European Classic'}]}

def load(tmp_path, records):
    p=tmp_path/'styles.json';p.write_text(json.dumps({'records':records}))
    return load_confirmed_paths(p)

@pytest.mark.parametrize('change',[
    {'level_1_id':'art_deco'},
    {'level_1_id':'modern'},
    {'level_2_id':'invented'},
    {'source_dataset':'hssd'},
    {'source_label':'invented'},
    {'confidence':1.1},
    {'evidence_type':'embedding_ranked'},
])
def test_invalid_confirmed_paths_fail_closed(tmp_path,change):
    r=record();r['styles'][0].update(change)
    with pytest.raises(ValueError):load(tmp_path,{AID:r})

def test_visual_cannot_claim_source_child(tmp_path):
    r=record();r['style_status']='visual_reviewed'
    with pytest.raises(ValueError):load(tmp_path,{AID:r})

def test_single_run_candidate_is_not_confirmed(tmp_path):
    r=record();r['style_status']='needs_review'
    assert load(tmp_path,{AID:r})=={}

def test_alias_conflict_rejected(tmp_path):
    r=record();other=copy.deepcopy(r);other['styles'][0].update(level_1_id='modern',level_2_id=None)
    with pytest.raises(ValueError):load(tmp_path,{AID:r,'hssd:'+AID:other})

def test_valid_source_and_alias(tmp_path):
    assert load(tmp_path,{'hssd:'+AID:record()})[AID]==[('traditional','european_classic','source_metadata')]

def test_hssd_x_encoded_id(tmp_path):
    aid='xxxx007297d3xb73ax40c8xa56ex51f170cfb73f'
    assert aid in load(tmp_path,{aid:record()})

def test_too_many_paths_rejected(tmp_path):
    r=record();r['styles']*=4
    with pytest.raises(ValueError):load(tmp_path,{AID:r})
