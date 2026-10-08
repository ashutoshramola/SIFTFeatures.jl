#!/usr/bin/env python3
"""Exercise descriptor-corruption controls using a scratch copy of the validator.
No project source or recorded validation result is modified. Requires the
optional validation Python dependencies and an instantiated Julia package.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / "validation/validate.py").read_text()
source = source.replace("ROOT = Path(__file__).resolve().parents[1]", f"ROOT = Path({str(root)!r})")
needle = '            jd = np.fromfile(work/(name+".desc.f32"),dtype=np.float32).reshape(-1,128)'
if source.count(needle) != 1:
    raise RuntimeError("Validator has changed; update this audit reproducer before using it.")
source = source.replace(needle, needle + '\n            jd[:] = 0  # Deliberate descriptor corruption in a scratch copy.')
with tempfile.TemporaryDirectory(prefix="sift-audit-guards-") as directory:
    scratch = Path(directory)
    script = scratch / "corrupted_validator.py"
    script.write_text(source)
    statuses = {}
    passed = {}
    for label, flags in (("normal", []), ("optimized", ["-O"])):
        output = scratch / f"{label}.json"
        result = subprocess.run([sys.executable, *flags, str(script), "--output", str(output)],
                                capture_output=True, text=True)
        statuses[label] = result.returncode
        print(f"{label}: exit={result.returncode}")
        if output.exists():
            report = json.loads(output.read_text())
            passed[label] = report["passed"]
            base = report["images"]["base"]
            print(f"  passed={report['passed']}, descriptor cosine median="
                  f"{base['detected_descriptor_cosine']['median']}, cross matches={base['cross_matches']}")
    if statuses["normal"] != 0 and statuses["optimized"] == 0:
        print("Confirmed: optimized Python bypasses the validation gates.")
        raise SystemExit(1)
    elif all(statuses[k] != 0 and passed.get(k) is False for k in ("normal", "optimized")):
        print("Protected: normal and optimized Python both reject corrupted descriptors and write failed reports.")
    else:
        raise RuntimeError("Unexpected control result; inspect exit statuses and reports.")
