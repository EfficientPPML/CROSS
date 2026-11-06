"""R7: original process/owner/Finalizer paths with declared recorded effects.

Admission, audit, backend/stage and signals are recordings installed before
snapshot capture. The default process owner factories and retirement functions
remain original. Tiny evidence files use the real Output owner; this module's
wrapper supplies explicit publication/close faults, never actual role evidence.
"""
from contextlib import contextmanager
import errno
import json
import os
import stat
import sys
from types import SimpleNamespace


_RUN_CASES = {
    "run_clean_retirement_order",
    "run_body_primary_cleanup_retention",
    "run_default_exit_failure70",
    "run_evidence_close_failure70",
    "run_swallowed_audit_denial70",
    "run_owner_accessor_cleanup_original",
}


def _parent(f, path):
    info = path.lstat()
    assert stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700
    return f.canonical(dict(path=str(path), device=info.st_dev, inode=info.st_ino, mode=0o700))


def _file_record(f, path, raw):
    return dict(path=path.relative_to(f.owned).as_posix(), bytes=len(raw),
                sha256=f.sha(raw), recording_fixture_only=True)


def _evidence_cap(f):
    output_module = f.module("keygen_output")
    parent = f.owned / "cap-parent"
    parent.mkdir(mode=0o700)
    owner = output_module.Output(str(parent / "evidence"), expected_parent=_parent(f, parent))
    original_fds = tuple(owner._owned_fds)
    assert len(original_fds) == 2 and len(set(original_fds)) == 2
    try:
        raw = b"{}\n"
        owner.write("report.json", raw)
        error = f.expect(
            (ValueError,), lambda: owner.write("failure.json", b"x" * ((2 << 20) + 1)),
            message="one bounded fresh evidence document required",
        )
        assert type(error) is ValueError
        assert not (owner.path / "failure.json").exists()
        assert sorted(p.name for p in owner.path.iterdir()) == ["report.json"]
        f.details["process_evidence_files"] = [_file_record(f, owner.path / "report.json", raw)]
    finally:
        owner.close()
    assert owner.fd is None and owner.parent_fd is None and owner._owned_fds == []
    for fd in original_fds:
        error = f.expect((OSError,), lambda: os.fstat(fd))
        assert error.errno == errno.EBADF
    f.no_device_work()
    return dict(passed=True, case="evidence_output_cap", group="R", recording_only=True,
                actual_authority=False, closed_original_descriptors=len(original_fds),
                checks=["real tiny owned evidence round-trip", "2MiB cap rejects before second publication",
                        "original directory descriptors retired"],
                scope="Real owned-file Output only; no server/device or application body.")


def _setup_run(f, case):
    f._graph()
    if case == "run_default_exit_failure70":
        f.mode = "exit_failure"
    f.native_seams()
    state = dict(outputs=[], error=None, swallowed=None, cleanup_marker=OSError("owned cleanup fault"),
                 runtime_retired=False, runtime_reentry=0, close_calls=0, original_close_calls=0,
                 snapshot=None, cleanup_report=None, records=[])
    original_output = f.module("keygen_output").Output
    # Only the two declared audit/public-key recordings consume this marker;
    # no genuine PublicProfile validation is inferred from the admitted graph.
    profile_marker = SimpleNamespace(recording_only=True)

    def admit(args, raw_map, expected, blobs):
        assert expected is f.expected and blobs is f.blobs
        assert raw_map == f.canonical(f.expected)
        f.event("R.admit.recording")
        return (f.packet, profile_marker, f.dep, f.res, {}, (), (),
                dict(recording_only=True, actual_capacity=False), f.device_packet_value)

    def authenticate_public(raw, profile, ids, *, role):
        assert raw is f.ready.public_manifest_bytes and ids is f.ids and role == "server"
        assert profile is profile_marker
        f.event("R.public_binding.recording")
        return f.key_public

    def bind_files(public, keys, identity):
        assert public is f.public and keys is f.key_public and json.loads(identity) == f.identity
        f.event("R.files.recording")
        return f.files

    @contextmanager
    def termination_signals():
        f.event("R.termination.enter")
        try:
            yield
        finally:
            f.event("R.termination.exit")

    @contextmanager
    def cleanup_signals():
        f.event("R.cleanup.enter")
        try:
            yield
        finally:
            f.event("R.cleanup.exit")

    class Backend:
        def __init__(self, runtime):
            f.runtime = runtime
            f.device = runtime.device_owner
            self.dependencies = SimpleNamespace(recording_only=True)
            self.factories = SimpleNamespace(recording_only=True)
            state["backend"] = self
            f.event("R.backend.recording")

    def stage(files, root, request, *, key_dependencies, factories):
        assert files is f.files and key_dependencies is state["backend"].dependencies
        assert factories is state["backend"].factories
        assert request == json.loads(f.phase.request_bytes)
        assert root == json.loads(f.phase.grant_bytes)["public_key_store"]["path"]
        assert f.audit.keys_enabled is True and f.audit.bodies_enabled is False
        f.event("R.stage.enter")
        try:
            f.transfers = tuple(f.array("R.setup." + str(i)) for i in range(4))
            f.device.register_loaded(f.handle, f.transfers)
            f.event("R.loaded.registered")
            f.device.post_keys(f.handle)
            f.runtime.admit_server_bodies(f.handle)
            f.full_owner = SimpleNamespace(recording_only=True)
            f.prepared = (f.ciphertext("R.prepared"),)
            f.device.before_evaluation(f.ctx, f.handle, f.full_owner, f.prepared)
            f.event("R.evaluation.boundary")
            if case == "run_body_primary_cleanup_retention":
                raise f.marker
            return b'{"recording_only":true}\n'
        finally:
            # Recorded loader cleanup is explicit; the original terminal owner
            # still checks this original handle before its terminal observation.
            f.handle.ctx = None
            f.event("R.loader.retired")

    class Evidence:
        def __init__(self, path, *, expected_parent):
            self.original = original_output(path, expected_parent=expected_parent)
            self.original_close = self.original.close
            state["outputs"].append(self)
            f.event("R.evidence.open")

        def check(self):
            return self.original.check()

        def write(self, name, raw):
            if case == "run_swallowed_audit_denial70" and name == "report.json":
                f.audit.denials.append("explicit recording publication denial")
                try:
                    f.audit.check()
                except PermissionError as error:
                    state["swallowed"] = error
                    f.event("R.audit.denial.swallowed")
            self.original.write(name, raw)
            state["records"].append(_file_record(f, self.original.path / name, raw))
            f.event("R.evidence.write:" + name)
            if case == "run_owner_accessor_cleanup_original" and name == "report.json":
                def replacement(owner):
                    f.counts["replacement"] += 1
                    raise AssertionError("replacement device accessor must not run")
                f.mutate(f.owner_module.DeviceOwner, "evidence", replacement)
                f.event("R.owner.accessor.mutated")

        def close(self, primary=None):
            state["close_calls"] += 1
            f.event("R.evidence.close.begin")
            try:
                state["original_close_calls"] += 1
                self.original_close(primary=primary)
            except BaseException as error:
                if case == "run_body_primary_cleanup_retention" and error is primary:
                    # The real owner has attempted its original descriptors.
                    # The outer original Finalizer must preserve the body
                    # identity even if this declared close seam raises another.
                    raise state["cleanup_marker"] from error
                raise
            finally:
                f.event("R.evidence.close.attempted")
            if case == "run_evidence_close_failure70":
                raise f.marker

    for owner, name, value, label in (
        (f.process, "admit", admit, "R: scientific admission recording and unvalidated profile marker"),
        (f.process, "AccessScope", f.Audit, "R: explicit audit effects; real audit tested in A"),
        (f.process, "hash_installed", f.installed_recording, "R: installed-source observation recording"),
        (f.process, "imported_origins", f.origins_recording, "R: backend-origin observation recording"),
        (f.process, "authenticate_public", authenticate_public, "R: public-key binding recording"),
        (f.process, "bind_files", bind_files, "R: declared file-admission recording"),
        (f.process, "termination_signals", termination_signals, "R: termination signal scope recording"),
        (f.process, "cleanup_signals", cleanup_signals, "R: cleanup signal scope recording"),
        (f.process, "Output", Evidence, "R: real owned Output with explicit publication/close fault wrapper"),
        (f.backend_module, "ServerBackend", Backend, "R: backend constructor effect recording"),
        (f.module("qwen_stage_composition"), "_server_for_stage", stage, "R: numerical stage effects with original owner transitions"),
    ):
        f.patch(owner, name, value, label)
    return state


def _invoke(f, state):
    previous = sys.getprofile()
    require_code = f.process.require_runtime.__code__
    owner_dispatch = f.owner_module._dispatch.__code__
    close_code = f.owner_module.close_default_scope.__code__
    capture_code = f.process._capture_snapshot.__code__
    run_code = f.process.run.__code__

    def observer(frame, event, arg):
        if previous is not None:
            previous(frame, event, arg)
        code = frame.f_code
        if event == "call":
            if code is require_code and state["runtime_retired"]:
                state["runtime_reentry"] += 1
            if code is owner_dispatch and frame.f_locals.get("action") == "retire":
                f.event("R.device.retire.begin")
            if code is close_code:
                f.event("R.default.close.original")
            if code.co_filename == f.process.__file__ and code.co_name == "retire":
                if f.runtime is not None and frame.f_locals.get("runtime") is f.runtime:
                    f.event("R.runtime.retire.begin")
                elif "retained" in frame.f_locals:
                    f.event("R.snapshot.retire.begin")
        elif event == "return":
            if code is capture_code:
                state["snapshot"] = frame.f_locals["retained"]
            if code is run_code:
                state["cleanup_report"] = frame.f_locals["cleanup_report"]
            if (code.co_filename == f.process.__file__ and code.co_name == "retire"
                    and f.runtime is not None and frame.f_locals.get("runtime") is f.runtime
                    and id(f.runtime) not in f.process._RUNTIME_CONTROLS):
                state["runtime_retired"] = True
                f.event("R.runtime.retired")

    args = SimpleNamespace(archive_sha256="a" * 64, packet_sha256=f.packet.expected_sha256,
                           device_packet_sha256=f.device_packet_value.expected_sha256)

    def invoke():
        try:
            return f.process.run(args, f.canonical(f.expected), f.expected, f.blobs,
                                 f.gate, f.entry.snapshot, f.entry.origins)
        except BaseException as error:
            state["error"] = error
            raise

    sys.setprofile(observer)
    try:
        return f.entry.terminal_status(invoke)
    finally:
        sys.setprofile(previous)


def run(f, case):
    if case == "evidence_output_cap":
        return _evidence_cap(f)
    assert case in _RUN_CASES
    state = _setup_run(f, case)
    status = _invoke(f, state)
    assert f.device is not None and f.runtime is not None
    row = f.owner_module._OWNERS[id(f.device)]
    assert row["owner"] is f.device and row["handle"] is f.handle and f.handle.ctx is None
    assert row["context"] is f.ctx and row["transfers"] is f.transfers
    assert state["snapshot"] is not None and state["cleanup_report"] is not None
    assert state["close_calls"] == state["original_close_calls"] == 1
    assert f.events.count("R.default.close.original") == 1 and f.events.count("audit.close") == 1
    assert f.events.count("R.termination.enter") == f.events.count("R.termination.exit") == 1
    assert f.events.count("R.cleanup.enter") == f.events.count("R.cleanup.exit") == 1
    assert state["runtime_reentry"] == 0
    assert all(output.original.fd is None and output.original.parent_fd is None for output in state["outputs"])
    f.details["process_evidence_files"] = state["records"]
    checks = ["original process/owner/runtime/Finalizer paths under declared recordings",
              "original evidence close and independent cleanup attempted exactly once",
              "no RuntimeAdmission reentry after its retirement"]

    if case == "run_clean_retirement_order":
        assert status == 0 and state["error"] is None
        ordered = ["R.runtime.retired", "R.device.retire.begin", "R.default.close.original",
                   "default.exit", "R.evidence.close.begin", "audit.close", "R.cleanup.exit",
                   "R.snapshot.retire.begin"]
        assert all(f.events.count(name) == 1 for name in ordered)
        assert [f.events.index(name) for name in ordered] == sorted(f.events.index(name) for name in ordered)
        assert row["mode"] == "retired" and state["snapshot"]["status"] == "retired"
        assert row["default_exit_attempted"] and state["runtime_retired"]
        assert f.process._QUARANTINE == []
        checks.append("runtime, device guard, original default scope, evidence/audit/signals, snapshot retire in order")
    else:
        assert status == 70 and state["error"] is not None
        assert f.process._QUARANTINE and f.process._QUARANTINE[-1][-1] is state["error"]
        assert state["snapshot"]["status"] != "retired"
        if case == "run_body_primary_cleanup_retention":
            assert state["error"] is f.marker
            assert any(item["type"] == "OSError" for item in state["cleanup_report"]["cleanup_errors"])
            assert row["failure"] is not None and row["quarantined"]
            checks.append("original body marker survives additional close fault; failed owner graph retained")
        elif case == "run_default_exit_failure70":
            assert state["error"] is f.jax.MARKER
            assert f.events.count("default.exit") == 1 and row["default_exit_attempted"]
            assert row["failure"] is not None and row["quarantined"] and state["runtime_retired"]
            checks.append("original default-scope exit fails once after runtime/device guard retirement; terminal70")
        elif case == "run_evidence_close_failure70":
            assert state["error"] is f.marker
            assert row["mode"] == "retired" and row["default_exit_attempted"] and state["runtime_retired"]
            checks.append("late evidence close gives70 while local runtime/device retirement remains accurately recorded")
        elif case == "run_swallowed_audit_denial70":
            assert isinstance(state["swallowed"], PermissionError)
            assert isinstance(state["error"], PermissionError) and f.audit.denials
            assert row["failure"] is not None and row["quarantined"]
            checks.append("swallowed publication denial remains a sticky process terminal veto")
        elif case == "run_owner_accessor_cleanup_original":
            assert f.counts["replacement"] == 0 and row["failure"] is not None and row["quarantined"]
            assert any(item["operation"] == "original device accessor identity"
                       for item in state["cleanup_report"]["cleanup_errors"])
            f.restore_mutations()
            f.expect((PermissionError,), f.device.check)
            assert f.counts["replacement"] == 0 and f.events.count("R.default.close.original") == 1
            checks.append("replaced accessor never invoked; original close attempted and failed owner stays failed")
    return dict(passed=True, case=case, group="R", terminal_status=status,
                recording_only=True, actual_authority=False, checks=checks,
                original_scope_exit_attempted=row["default_exit_attempted"],
                local_device_mode=row["mode"], snapshot_status=state["snapshot"]["status"],
                retained_process_failure=bool(f.process._QUARANTINE),
                ordered_events=[event for event in f.events if event.startswith("R.") or event in ("default.exit", "audit.close")],
                scope="Original orchestration over explicit recordings and tiny owned evidence; no actual server or HE result.")
