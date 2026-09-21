import json
from pathlib import Path
import pytest
from scripts.style_embedding_v2 import canonical_asset_uid,load_confirmed_paths
from scripts.prepare_all_asset_embeddings import content
from scripts.index_all_asset_embeddings import document_id,prompt

def test_source_uid_no_collision():
    ids=['3dfuture:abc','others:abo:abc','others:gso:abc','generated:hunyuan:abc']
    assert len({document_id(x) for x in ids})==4
    assert all(len(document_id(x))==64 and document_id(x).isalnum() for x in ids)

def test_external_case_preserved():
    assert canonical_asset_uid('others:abo:B002CM3J2A')=='others:abo:B002CM3J2A'
    assert canonical_asset_uid('a'*40)=='hssd:'+'a'*40

def test_3dfuture_source_style_injected(tmp_path):
    r={'schema_version':'asset_style@2.0','style_status':'source_metadata',
       'provenance':{'ontology_id':'bonn_furniture_styles_hierarchical@1.0'},
       'styles':[{'level_1_id':'traditional','level_2_id':'european_classic',
                  'source_dataset':'3dfuture','source_label':'European Classic',
                  'evidence_type':'source_metadata','confidence':.98}]}
    p=tmp_path/'style.json';p.write_text(json.dumps({'3dfuture:abc':r}))
    paths=load_confirmed_paths(p,all_sources=True)['3dfuture:abc']
    text=content('3dfuture:abc','3dfuture','Floor Lamp',paths)
    assert 'traditional > european classic' in text
    assert 'HSSD asset' not in text
    assert load_confirmed_paths(p)=={}

def test_unstyled_not_default_modern():
    text=content('others:gso:abc','gso','cup',[])
    assert 'Verified styles' not in text and 'modern' not in text

def test_multimodal_markers_match_views():
    assert prompt({'content':'Asset UID: others:abo:abc.','image_paths':['a','b']}).count('<__media__>')==2

def test_pathlike_uid_rejected():
    with pytest.raises(ValueError):canonical_asset_uid('others:abo:../x')

def test_request_rejects_image_mutation(tmp_path):
    from scripts.index_all_asset_embeddings import embed_checked
    image=tmp_path/'input.png';image.write_bytes(b'before')
    class ChangingClient:
        def embed_images_text(self,images,text):
            images[0].write_bytes(b'after')
            return [1.0]
    row={'asset_uid':'others:abo:test','content':'test','image_paths':[str(image)]}
    with pytest.raises(ValueError,match='images changed during request'):
        embed_checked(ChangingClient(),row,[image])

def test_batch_checkpoint_only_after_flush(tmp_path,monkeypatch):
    from scripts import index_all_asset_embeddings as m
    ledger=tmp_path/'done.jsonl';ledger.write_text('{"asset_uid":"old"}\n')
    calls=[]
    monkeypatch.setattr(m.base,'flush_docs',lambda c,d:calls.append(('upsert',len(d))))
    class Collection:
        def flush(self):
            calls.append(('flush',))
            assert ledger.read_text()=='{"asset_uid":"old"}\n'
    m.commit_batch(Collection(),[1,2],[{'asset_uid':'a'},{'asset_uid':'b'}],ledger)
    assert calls==[('upsert',2),('flush',)]
    assert [json.loads(x)['asset_uid'] for x in ledger.read_text().splitlines()]==['old','a','b']

def test_failed_flush_does_not_checkpoint(tmp_path,monkeypatch):
    from scripts import index_all_asset_embeddings as m
    ledger=tmp_path/'done.jsonl'
    monkeypatch.setattr(m.base,'flush_docs',lambda c,d:None)
    class Collection:
        def flush(self):raise RuntimeError('disk failure')
    with pytest.raises(RuntimeError,match='disk failure'):
        m.commit_batch(Collection(),[1],[{'asset_uid':'a'}],ledger)
    assert not ledger.exists()

def test_resume_audit_detects_changed_images(tmp_path):
    from scripts.index_all_asset_embeddings import verify_completed_input
    import hashlib
    asset=tmp_path/'model.obj';asset.write_text('model')
    image=tmp_path/'view.png';image.write_bytes(b'original')
    row={'asset_uid':'test','asset_path':str(asset),'image_paths':[str(image)]}
    hashes=[hashlib.sha256(b'original').hexdigest()]
    verify_completed_input((row,hashes))
    image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='completed input images changed'):
        verify_completed_input((row,hashes))
