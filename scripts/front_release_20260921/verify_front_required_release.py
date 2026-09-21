"""Verify actual consumer reads and whole-inventory front coverage after publication."""
import importlib.util, json, math, sys, time
import front_required_release as p

def main():
    samples={};methods={};total=0
    for index,root in enumerate(p.ROOTS):
        manifest=p.load(root/p.MANIFEST);count=0
        overlay=p.load(root/'canonical_front_required.json.gz')
        assert p.sha(root/'canonical_front_required.json.gz')==manifest['front_overlay_sha256']
        for name,h in manifest['sha256'].items():
            assert p.sha(root/name)==h,(str(root),name)
            rows=p.load(root/name)
            for key,row in rows.items():
                uid=p.uid(key,row,name);front=row['canonical_front'];v=front['canonical_orientation_axis']
                assert len(v)==3 and all(math.isfinite(x) for x in v)
                assert abs(sum(x*x for x in v)-1)<1e-6 and v[1]==0
                assert front['asset_local_front_axis']==v and front==overlay[uid]
                assert front['canonical_front_direction'] and front['canonical_orientation_source']
                c=row.get('front_visual_candidate')
                if c and c['result']['front_status'] in ['uncertain','unusable','no_unique']:
                    assert front['canonical_orientation_is_semantic_front'] is False
                if c and front['canonical_orientation_source']=='luna_v2_unique':
                    N=c['source_to_evidence_matrix'];native=c['front_axis_source_local']
                    evidence=[sum(N[i][j]*native[j] for j in range(3)) for i in range(3)]
                    assert all(abs(a-b)<1e-6 for a,b in zip(evidence,v)),uid
                source=uid.split(':')[0];samples.setdefault(source,uid);count+=1
            del rows
        assert count==manifest['total']==len(overlay)
        total=count
    sys.path.insert(0,str(p.REPO))
    from hssd_asset_library import AssetLibrary
    for root in [p.LOCAL,p.REPO/'data']:
        lib=AssetLibrary(root,sources='all')
        for source,uid in samples.items():
            if source!='generated':
                rec=lib.require(uid)
                assert rec['canonical_front']==overlay[uid]
                rec=lib.require_unified(uid)
                assert rec['canonical_front']==overlay[uid]
        del lib
    spec=importlib.util.spec_from_file_location('reader',p.TASK/'scenesmith/scenebenchmark_critic/current_asset_annotations.py')
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    for uid in samples.values():assert reader.get_current_asset_annotation(uid)['canonical_front']==overlay[uid]
    style=p.load(p.SHARED/'styles/CURRENT_STYLE_RELEASE.json')
    assert p.sha(p.Path(style['overlay']))==style['overlay_sha256']
    output={'passed':True,'verified_at':time.time(),'total':total,'roots_verified':list(map(str,p.ROOTS)),
        'front_nonempty_unit_horizontal':total,'consumer_samples':samples,'source_frame_transform_verified':True,
        'style_overlay_unchanged':True,'task32_full_package_test':'not claimed: CPU environment has no pydrake'}
    p.write(p.OUT/'VERIFIED.json',output);print(json.dumps(output,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
