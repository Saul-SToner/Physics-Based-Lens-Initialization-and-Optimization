"""Continue the validated random stream until the requested number passes."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import exp, log
from pathlib import Path
from random import Random

from generate_random_prescriptions import RANGES, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    if args.output.exists() or args.count <= 0:
        parser.error("Output must be new and count positive")
    rng = Random(args.seed)
    candidates, rejected = [], []
    attempt_id = 0
    def draw(name):
        low, high = RANGES[name]
        return exp(rng.uniform(log(low), log(high)))
    while len(candidates) < args.count:
        attempt_id += 1
        if attempt_id > args.count*100:
            raise RuntimeError("Sampling attempt limit")
        object_distance = draw("object_distance_magnitude_mm")
        image = draw("image_distance_mm")
        entrance = object_distance*draw("entrance_pupil_fraction_of_object_distance")
        exit_pupil = draw("exit_pupil_distance_mm")
        magnification = draw("magnification_magnitude")
        row = evaluate((-object_distance, image, -entrance, exit_pupil, -magnification))
        row["sampling_attempt_id"] = attempt_id
        if row["passed"]:
            row["id"] = len(candidates)+1
            candidates.append(row)
        else:
            rejected.append(row)
    root = Path(__file__).resolve().parent
    original = json.loads((root/"random_candidates_10/candidates.json").read_text())["candidates"]
    if args.seed == 20261118:
        assert all(a["inputs"] == list(b["inputs"]) for a,b in zip(original,candidates))
    sources = ("generate_batch_candidates.py","generate_random_prescriptions.py","two_lens.py",
               "lens_geometry.py","validate_two_lens.py","compare_opticstudio.py")
    report = {
        "seed":args.seed,"count":args.count,"sampling_ranges":RANGES,
        "sampling_policy":"continue deterministic log-uniform stream; retain all rejected candidates separately",
        "source_sha256":{name:sha256((root/name).read_bytes()).hexdigest() for name in sources},
        "summary":{"attempts":attempt_id,"success_count":len(candidates),"failure_count":len(rejected),
                   "success_rate":len(candidates)/attempt_id,
                   "failure_reasons":dict(Counter(reason for row in rejected for reason in row["failure_reasons"]))},
        "candidates":candidates,"rejected":rejected,
    }
    args.output.mkdir(parents=True)
    (args.output/"candidates.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report["summary"],indent=2))


if __name__ == "__main__":
    main()
