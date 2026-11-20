"""Cross starting prescriptions and DLS budgets at the same aperture and field."""

import argparse
import json
from pathlib import Path

from optimize_opticstudio_rules import TARGET_F, TARGET_FIELD, attempt, load_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("rule_1", type=Path)
    parser.add_argument("rule_2", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import clr

    for assembly in ("ZOSAPI_Interfaces", "ZOSAPI"):
        clr.AddReference(str(args.installation.resolve() / f"{assembly}.dll"))
    import ZOSAPI as zos

    connection = zos.ZOSAPI_Connection()
    application = connection.CreateNewApplication()
    if application is None:
        raise RuntimeError("Could not start an OpticStudio standalone session")
    try:
        if not application.IsValidLicenseForAPI:
            raise RuntimeError(f"ZOS-API license unavailable: {application.LicenseStatus}")
        system = application.PrimarySystem
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        results = []
        for name, seed in (("C", args.baseline), ("rule_1", args.rule_1), ("rule_2", args.rule_2)):
            for cycles in (5, 50):
                load_seed(system, seed.resolve())
                result = attempt(zos, system, TARGET_F, TARGET_FIELD, cycles)
                path = output / f"{name}_{cycles}_cycles.zmx"
                system.SaveAs(str(path))
                results.append({
                    "seed": name, "dls_cycles": cycles, "prescription": path.name, **result,
                })
                print(name, cycles, "passed" if result["accepted"] else "failed", flush=True)
        (output / "target_comparison.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    finally:
        application.CloseApplication()


if __name__ == "__main__":
    main()
