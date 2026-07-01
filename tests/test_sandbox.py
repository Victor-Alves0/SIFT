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
