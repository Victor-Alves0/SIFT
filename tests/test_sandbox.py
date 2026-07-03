"""SubprocessSandbox: isolated execution with tool-call proxy + limits.

Spawns a real child process (python -m sift._sandbox_child), so these are a bit
slower than the in-process tests but exercise the real isolation path.
"""
import json

from sift import Sift
from sift.sandbox import SubprocessSandbox


def _sift(sandbox) -> Sift:
    s = Sift(retrieval="bm25", sandbox=sandbox)

    @s.tool("mail.gmail.read", description="Read emails from the inbox",
            params={}, returns=["id", "subject"])
    def _r():
        return {"id": "1", "subject": "Hi", "secret": "hidden"}

    s.build_index()
    return s


def test_subprocess_composes_and_filters():
    s = _sift(SubprocessSandbox(timeout=20))
    out = json.loads(s.run_code(
        "m = call('mail.gmail.read')\noutput = {'subj': m['subject'], 'keys': sorted(m)}"))
    assert out["output"]["subj"] == "Hi"
    assert out["output"]["keys"] == ["id", "subject"]  # 'secret' filtered by the parent


def test_subprocess_search_proxy():
    s = _sift(SubprocessSandbox(timeout=20))
    out = json.loads(s.run_code("paths = search('read emails inbox')\noutput = {'n': len(paths)}"))
    assert out["output"]["n"] >= 1


def test_subprocess_blocks_imports():
    s = _sift(SubprocessSandbox(timeout=20))
    out = json.loads(s.run_code("import os\noutput = 1"))
    assert "error" in out and "import" in out["error"].lower()


def test_subprocess_line_budget():
    s = _sift(SubprocessSandbox(timeout=20, max_lines=5000))
    out = json.loads(s.run_code("while True:\n    x = 1"))
    assert "error" in out and "budget" in out["error"].lower()


def test_subprocess_wallclock_timeout():
    # a single-line C-level loop the line budget can't see -> watchdog kills it
    s = _sift(SubprocessSandbox(timeout=2))
    out = json.loads(s.run_code("output = sum(range(1000000000))"))
    assert "error" in out and "timeout" in out["error"].lower()


def test_child_is_lightweight():
    """The sandbox child must not import the sift package (nor numpy) — it loads
    sandbox.py standalone. A fat child doubles run_code's boot cost and widens
    the surface of the process that runs untrusted code."""
    import subprocess
    import sys
    from pathlib import Path

    import sift

    child = Path(sift.__file__).with_name("_sandbox_child.py")
    probe = (
        "import sys, runpy\n"
        f"runpy.run_path(r'{child}', run_name='not_main')\n"   # module body only
        "print('sift' in sys.modules, 'numpy' in sys.modules)\n"
    )
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["False", "False"], r.stdout


def test_scope_run_code_respects_result_cap():
    """dispatch('run_code') through a SCOPE is capped like the Sift path."""
    s = Sift(retrieval="bm25", max_result_chars=200)

    @s.tool("data.blob.get", description="Get a huge blob", params={}, returns=[])
    def _b():
        return {"data": "x" * 2000}

    s.build_index()
    view = s.scope(allow=["data.*"])
    out = view.dispatch("run_code", {"code": "output = call('data.blob.get')['data']"})
    assert len(out) < 400 and "truncated" in out
