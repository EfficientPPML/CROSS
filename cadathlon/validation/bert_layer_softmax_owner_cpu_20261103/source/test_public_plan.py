import json
from pathlib import Path
from public_plan import build

def test_all1536_schedules_and4608_actual_scale_paths():
    report=build();assert report['public_cases']==1536 and len(report['layers'])==12
    assert len(report['graph_keys'])==14 and set(report['graph_keys'])<=set(report['bootstrap_keys'])
    for layer,entry in report['layers'].items():
        assert entry['continuation_after_final_mask'] in (17,18)
        assert entry['selected_squarings']==[5,7,7,7,6,7,7,6,6,6,6,6][int(layer)]
        for key,case in entry['cases'].items():
            assert len(case['scale_variants'])==3
            for variant in case['scale_variants'].values():
                assert variant['initial_output_level']==(23 if layer=='0' else 6)
                assert variant['final_mask_output_level']==24
                assert all(row['output_level']>=2 for row in variant['rounds'])
                assert variant['explicit_B2_refresh_count']>=1
    Path(__file__).with_name('public_plan.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
