"""
Run all available OOF exports, cleanly skipping those that fail (missing data,
missing dependency, missing checkpoint).

    python -m analysis.exporters.run_all
"""
import importlib
import traceback

EXPORTERS = [
    "export_dynamic_irt",
    "export_girt",
    "export_bkt",
    "export_pfa",
    "export_optimized_pfa",
    "export_gkt",
    "export_irt",
]

if __name__ == "__main__":
    for name in EXPORTERS:
        print(f"\n{'='*60}\n  {name}\n{'='*60}")
        try:
            mod = importlib.import_module(f"analysis.exporters.{name}")
            mod.main()
        except Exception as e:
            print(f"  [failed] {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=1)