"""Run alternating aperture and field continuation from the frozen C file."""

import argparse
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path

from optimize_opticstudio_rules import (
    START_F, START_FIELD, TARGET_F, TARGET_FIELD, attempt, load_seed,
)


def alternate(zos, system, baseline, destination, cycles, glass_edge_target_mm):
    load_seed(system, baseline)
    current = {"aperture": START_F, "field": START_FIELD}
    targets = {"aperture": TARGET_F, "field": TARGET_FIELD}
    factors = {"aperture": 0.707, "field": 1.414}
    history = []

    def try_candidate(settings, stage):
        result = attempt(
            zos, system, settings["aperture"], settings["field"], cycles, glass_edge_target_mm,
        )
        result["stage"] = stage
        history.append(result)
        print(stage, settings, "accepted" if result["accepted"] else "rejected", flush=True)
        if result["accepted"]:
            system.SaveAs(str(destination))
        return result["accepted"]

    stopped = {}
    if try_candidate(current, "initial"):
        while len(stopped) < 2:
            for dimension in ("aperture", "field"):
                if dimension in stopped:
                    continue
                factor = factors[dimension]
                aperture = dimension == "aperture"
                target = targets[dimension]
                candidate = dict(current)
                proposed = current[dimension] * factor
                candidate[dimension] = max(target, proposed) if aperture else min(target, proposed)
                if try_candidate(candidate, dimension):
                    current = candidate
                    if current[dimension] == target:
                        stopped[dimension] = "target_reached"
                else:
                    load_seed(system, destination)
                    factor = sqrt(factor)
                    factors[dimension] = factor
                    if (aperture and factor > 0.99) or (not aperture and factor < 1.01):
                        stopped[dimension] = "step_limit"
    else:
        stopped = dict.fromkeys(targets, "initial_rejected")
    return {
        "order": ["aperture", "field"],
        "dls_cycles_per_step": cycles,
        "glass_edge_target_mm": glass_edge_target_mm,
        "target_reached": current == targets,
        "stop_reasons": stopped,
        "final_accepted": next((row for row in reversed(history) if row["accepted"]), None),
        "steps": history,
    }


def verify_freeze(manifest, hashes):
    for relative, expected in hashes.items():
        if sha256((manifest.parent / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen file changed: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation", type=Path)
    parser.add_argument("freeze_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = args.freeze_manifest.resolve()
    freeze = json.loads(manifest.read_text(encoding="utf-8"))
    settings = freeze["settings"]
    verify_freeze(manifest, freeze["sha256"])
    baseline = manifest.parent / "opticstudio_comparison/C_restored_conjugates.zmx"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

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
        result = alternate(
            zos, application.PrimarySystem, baseline, output / "rule_3.zmx",
            settings["dls_cycles_per_step"], settings["glass_edge_target_mm"],
        )
        verify_freeze(manifest, freeze["sha256"])
        (output / "rules.json").write_text(
            json.dumps({"rule_3": result}, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
    finally:
        application.CloseApplication()


if __name__ == "__main__":
    main()
