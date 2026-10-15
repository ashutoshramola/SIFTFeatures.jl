"""Validation gates and durable reports; checks also run under python -O."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                    prefix="." + path.name, delete=False)
    temporary = Path(f.name)
    try:
        with f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finite_report(value):
    """Preserve diagnostics when a failed calculation produced NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_report(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_report(v) for v in value]
    return value


class Gates:
    def __init__(self, report):
        self.report = report
        report["checks"] = []
        report["failures"] = []

    def require(self, condition, name, detail=""):
        passed = bool(condition)
        check = {"name": name, "passed": passed, "detail": str(detail)}
        self.report["checks"].append(check)
        if not passed:
            self.report["failures"].append(check)
        return passed


def execute(args, runner, suite):
    sources = [ROOT / "Project.toml", *sorted((ROOT / "src").glob("*.jl")),
               *sorted((ROOT / "validation").glob("*.py")),
               *sorted((ROOT / "validation").glob("*.jl"))]
    report = {"suite": suite, "run_id": uuid.uuid4().hex,
              "started_utc": now(), "state": "running", "passed": False,
              "python": sys.version.split()[0], "python_optimization": sys.flags.optimize,
              "platform": platform.platform(),
              "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources}}
    gates = Gates(report)
    # Replace an old success before starting work, including if Julia later fails.
    write_report(args.output, report)
    try:
        runner(args, report, gates)
        json.dumps(report, allow_nan=False)
        report["state"] = "complete"
        report["passed"] = not report["failures"]
    except Exception as error:
        report["passed"] = False
        report["state"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        report = finite_report(report)
        print(report["error"], file=sys.stderr, flush=True)
    report["finished_utc"] = now()
    write_report(args.output, report)
    print(f"{suite}: {'PASS' if report['passed'] else 'FAIL'}; "
          f"{len(report['failures'])} failed checks. Report: {args.output}", flush=True)
    return 0 if report["passed"] else 1
