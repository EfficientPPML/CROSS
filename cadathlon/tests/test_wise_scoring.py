"""Incomplete, mismatched or invalid measurements cannot earn P5 speed credit."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest

path = Path(__file__).resolve().parents[1] / "agent_pset/p5_wise/src/score.py"
spec = importlib.util.spec_from_file_location("wise_scoring", path)
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


def report(backend):
    sample = {"server_s": 2. if backend == "cpu" else 1., "client_to_client_s": 3.,
              "passed": True, "quality": {"max_abs_error": 1e-7, "top1_agreement": 1.}}
    return {"status": "completed", "passed": True, "mode": "HE-WISE-NN", "backend": backend,
            "warmups": 1, "repeats": 3, "sources_before": {"runner": "abc"}, "sources_after": {"runner": "abc"},
            "case": {"batch": 8}, "crypto_profile": {"security_bits": 128, "degree": 8192},
            "schedule": {"sha256": "fixed"}, "samples": [copy.deepcopy(sample) for _ in range(3)],
            "warmup_samples": [copy.deepcopy(sample)]}


def test_recomputes_speed_from_raw_samples():
    cpu, tpu = report("cpu"), report("tpu")
    tpu["median_server_s"] = .00001
    assert scoring.compare(cpu, tpu)["speedup"] == 2


@pytest.mark.parametrize("field,value", [("status", "running"), ("passed", False), ("mode", "plaintext-reference"),
                                        ("backend", "cpu"), ("warmups", 0), ("repeats", 2),
                                        ("sources_after", {}), ("samples", [])])
def test_rejects_incomplete_or_wrong_scope(field, value):
    tpu = report("tpu"); tpu[field] = value
    with pytest.raises(ValueError): scoring.compare(report("cpu"), tpu)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_rejects_invalid_raw_timing(bad):
    tpu = report("tpu"); tpu["samples"][1]["server_s"] = bad
    with pytest.raises(ValueError): scoring.compare(report("cpu"), tpu)


def test_rejects_failed_warmup_and_mismatched_workload():
    for mutation in (lambda t: t["warmup_samples"][0]["quality"].update(max_abs_error=.1),
                     lambda t: t["case"].update(batch=1),
                     lambda t: t["crypto_profile"].update(degree=4096),
                     lambda t: t["schedule"].update(sha256="different")):
        tpu = report("tpu"); mutation(tpu)
        with pytest.raises(ValueError): scoring.compare(report("cpu"), tpu)


def native_reports():
    import sys
    sys.path.insert(0, str(path.parent))
    from wise_source import WISE_COMMIT
    cpu, tpu = report("cpu"), report("tpu")
    cpu.pop("crypto_profile")
    cpu.update(engine="WISE/OpenFHE", profile={"security_level": "HEStd_128_classic", "degree": 16384},
               native_binary_hashes_before={"fhecore.so": "fixed", "libOPENFHE.so": "fixed"},
               native_binary_hashes_after={"fhecore.so": "fixed", "libOPENFHE.so": "fixed"})
    for r in [cpu, tpu]:
        r["schedule"]["wise"] = {"commit": WISE_COMMIT, "source_sha256": {"planner": "fixed"}}
    cpu["schedule"]["sha256"] = "different-secure-cpu-packing"
    return cpu, tpu


def test_native_comparison_permits_separate_secure_packing_without_relabeling_hardware_speedup():
    cpu, tpu = native_reports()
    assert scoring.compare_native(cpu, tpu)["speedup_over_native_cpu"] == 2
    with pytest.raises(ValueError): scoring.compare(cpu, tpu)


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(mode="client-layer-diagnostic"),
    lambda c: c["profile"].update(security_level="HEStd_NotSet"),
    lambda c: c.update(native_binary_hashes_after={}),
    lambda c: c["case"].update(batch=1),
    lambda c: c["schedule"]["wise"].update(commit="unreviewed"),
    lambda c: c["schedule"]["wise"].update(source_sha256={}),
    lambda c: c["samples"][0]["quality"].update(max_abs_error=.01),
])
def test_native_comparison_rejects_diagnostics_changed_workloads_binaries_or_failed_accuracy(mutation):
    cpu, tpu = native_reports()
    mutation(cpu)
    with pytest.raises(ValueError): scoring.compare_native(cpu, tpu)


def pretrained_reports():
    cpu, tpu = native_reports()
    from pretrained_mnist import ASSETS
    for r in (cpu, tpu):
        r.update(case={"network": "wise_mnist", "batch": 8}, pretrained_checkpoint_evaluated=True, samples_scored=8)
        r['schedule']['wise']['asset_sha256'] = dict(ASSETS)
        for s in r["samples"] + r["warmup_samples"]:
            s["quality"].update(max_abs_error=2e-4, nrmse=3e-5)
    return cpu, tpu


def test_pretrained_comparison_enforces_its_distinct_accuracy_and_sample_contract():
    cpu, tpu = pretrained_reports()
    assert scoring.compare_native(cpu, tpu)["speedup_over_native_cpu"] == 2


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(pretrained_checkpoint_evaluated=False),
    lambda r: r.update(samples_scored=1),
    lambda r: r["samples"][0]["quality"].update(nrmse=.001),
    lambda r: r["samples"][0]["quality"].pop("nrmse"),
    lambda r: r["warmup_samples"][0]["quality"].update(max_abs_error=.002),
    lambda r: r['schedule']['wise'].pop('asset_sha256'),
    lambda r: r['schedule']['wise'].update(asset_sha256={}),
    lambda r: r['schedule']['wise']['asset_sha256'].update({'clear/mnist/HerPN.py': 'changed'}),
])
def test_pretrained_comparison_rejects_incomplete_or_failed_results(mutation):
    cpu, tpu = pretrained_reports()
    mutation(tpu)
    with pytest.raises(ValueError): scoring.compare_native(cpu, tpu)


def invoke_cli(tmp_path, monkeypatch, capsys, cpu_reports, tpu_reports, *, native=True, pretrained=False):
    cpu_paths, tpu_paths = [], []
    for i, (cpu, tpu) in enumerate(zip(cpu_reports, tpu_reports)):
        for label, report_value, paths in [('cpu', cpu, cpu_paths), ('tpu', tpu, tpu_paths)]:
            output = tmp_path / f'{label}{i}.json'
            output.write_text(json.dumps(report_value))
            paths.append(str(output))
    argv = ['score.py', *(['--native-only', '--native-cpu'] if native else ['--cpu']),
            *cpu_paths, '--tpu', *tpu_paths, *(['--pretrained'] if pretrained else [])]
    monkeypatch.setattr(sys, 'argv', argv)
    scoring.main()
    return json.loads(capsys.readouterr().out)


def six_reports(native=True):
    cpu_reports, tpu_reports = [], []
    for fixture in sorted((path.parent.parent / 'test').glob('test*.in')):
        cpu, tpu = native_reports() if native else (report('cpu'), report('tpu'))
        for value in (cpu, tpu):
            value['case'] = json.loads(fixture.read_text())
        cpu_reports.append(cpu)
        tpu_reports.append(tpu)
    return cpu_reports, tpu_reports


def test_native_only_cli_reports_complete_six_without_cross_or_contest_points(tmp_path, monkeypatch, capsys):
    cpu, tpu = six_reports()
    result = invoke_cli(tmp_path, monkeypatch, capsys, cpu, tpu)
    assert len(result['native_cpu_comparison']) == 6
    assert result['geomean_speedup_over_native_cpu'] == 2
    assert result['contest_scoring_eligible'] is False
    assert 'geomean_speedup' not in result and 'cases' not in result
    assert not any(name.endswith('_points') for name in result)


def test_matched_cross_cli_retains_existing_contest_scoring(tmp_path, monkeypatch, capsys):
    cpu, tpu = six_reports(native=False)
    result = invoke_cli(tmp_path, monkeypatch, capsys, cpu, tpu, native=False)
    assert result['geomean_speedup'] == 2
    assert result['correctness_points'] == 40 and result['performance_points'] == 20
    assert 'native_cpu_comparison' not in result


def test_native_only_pretrained_cli_preserves_slower_tpu_results(tmp_path, monkeypatch, capsys):
    cpu, tpu = pretrained_reports()
    for sample in tpu['warmup_samples'] + tpu['samples']:
        sample['server_s'] = 4.
    result = invoke_cli(tmp_path, monkeypatch, capsys, [cpu], [tpu], pretrained=True)
    assert result['scope'] == 'pretrained MNIST batches'
    assert result['geomean_speedup_over_native_cpu'] == .5
    assert result['native_cpu_comparison'][0]['speedup_over_native_cpu'] == .5
    assert not any(name.endswith('_points') for name in result)


@pytest.mark.parametrize('mutation', [
    lambda cpu, tpu: tpu.update(status='running'),
    lambda cpu, tpu: cpu.update(native_binary_hashes_after={}),
    lambda cpu, tpu: tpu.update(samples_scored=1),
    lambda cpu, tpu: tpu['warmup_samples'][0].update(passed=False),
])
def test_native_only_cli_cannot_bypass_measurement_gates(tmp_path, monkeypatch, capsys, mutation):
    cpu, tpu = pretrained_reports()
    mutation(cpu, tpu)
    with pytest.raises(ValueError):
        invoke_cli(tmp_path, monkeypatch, capsys, [cpu], [tpu], pretrained=True)
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('duplicate', [False, True])
def test_native_only_synthetic_cli_requires_all_six_distinct_cases(tmp_path, monkeypatch, capsys, duplicate):
    cpu, tpu = six_reports()
    if duplicate:
        cpu[-1], tpu[-1] = copy.deepcopy(cpu[0]), copy.deepcopy(tpu[0])
    else:
        cpu.pop(); tpu.pop()
    with pytest.raises(ValueError if duplicate else SystemExit):
        invoke_cli(tmp_path, monkeypatch, capsys, cpu, tpu)
    assert capsys.readouterr().out == ''


def test_native_only_cli_rejects_duplicate_pretrained_batches(tmp_path, monkeypatch, capsys):
    cpu, tpu = pretrained_reports()
    with pytest.raises(ValueError, match='distinct complete MNIST batches'):
        invoke_cli(tmp_path, monkeypatch, capsys, [cpu, cpu], [tpu, tpu], pretrained=True)


@pytest.mark.parametrize('args', [
    ['--native-only', '--tpu', 'missing.json'],
    ['--native-only', '--native-cpu', 'missing.json', '--cpu', 'missing.json', '--tpu', 'missing.json'],
    ['--native-cpu', 'missing.json', '--tpu', 'missing.json'],
    ['--native-only', '--native-cpu', 'a.json', 'b.json', '--tpu', 'c.json'],
])
def test_native_only_cli_requires_explicit_unambiguous_mode(monkeypatch, capsys, args):
    monkeypatch.setattr(sys, 'argv', ['score.py', *args])
    with pytest.raises(SystemExit):
        scoring.main()
    assert capsys.readouterr().out == ''
