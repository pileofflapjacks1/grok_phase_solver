"""Local numpy test runner (no pytest required)."""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

mods = [
    "tests.test_unique_asu",
    "tests.test_trial_res_symm",
    "tests.test_peak_budget",
    "tests.test_q_peaks_handbuild",
]
failed = 0
passed = 0
for modname in mods:
    mod = __import__(modname, fromlist=["*"])
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        fn = getattr(mod, name)
        try:
            fn()
            print("PASS", modname, name)
            passed += 1
        except Exception:
            print("FAIL", modname, name)
            traceback.print_exc()
            failed += 1
print("%d passed, %d failed" % (passed, failed))
sys.exit(1 if failed else 0)
