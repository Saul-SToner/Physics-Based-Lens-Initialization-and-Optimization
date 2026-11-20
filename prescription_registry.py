"""Lossless prescription identity and append-only cross-batch registration."""

from fractions import Fraction
from hashlib import sha256
import json
from math import isfinite, nextafter
import sqlite3


INPUT_NAMES = ("object_l1_mm", "image_l2_mm", "entrance_pupil_l1_mm",
               "exit_pupil_l2_mm", "magnification")
FIXED = {
    "index_nd":1.5, "abbe_vd":60.0, "dpgf":0.0,
    "wavelength_mm":0.000587, "center_thicknesses_mm":[1.0,1.0],
    "semi_apertures_mm":[1.0,1.0], "initial_working_f_number":500.0,
    "initial_fields_deg":[0.0,0.707,1.0],
    "shape":"symmetric spherical elements",
    "reference_planes":"lens vertex midpoints",
    "medium":"air",
}


def canonical(value):
    if isinstance(value, dict):
        return {key:canonical(item) for key,item in sorted(value.items())}
    if isinstance(value, (list,tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    number = float(value)
    if not isfinite(number):
        raise ValueError("Nonfinite value in prescription")
    return (0.0 if number == 0 else number).hex()


def fingerprint(parameters, schema):
    payload = json.dumps({"schema":schema, "parameters":canonical(parameters)},
                         sort_keys=True, separators=(",",":"), ensure_ascii=True)
    return sha256(payload.encode("ascii")).hexdigest(), payload


def sampled_parameters(candidate):
    if len(candidate["inputs"]) != len(INPUT_NAMES):
        raise ValueError("Expected five sampled parameters")
    return {"sampled":dict(zip(INPUT_NAMES,candidate["inputs"])), "fixed":FIXED}


def sample_fingerprint(candidate):
    return fingerprint(sampled_parameters(candidate), "two-lens-sampling-binary64-v1")


def geometry_parameters(candidate):
    entity = candidate["entity"]
    return {"radii_mm":entity["radii_mm"],
            "center_thicknesses_mm":entity["center_thicknesses_mm"],
            "air_gap_mm":entity["air_gap_mm"], "index_nd":entity["index"],
            "semi_apertures_mm":entity["semi_apertures_mm"],
            "abbe_vd":60.0,"dpgf":0.0}


def full_parameters(candidate):
    entity = candidate["entity"]
    return {**sampled_parameters(candidate), "geometry":geometry_parameters(candidate),
            "conjugate_vertex_distances_mm":{
                name:entity[name] for name in (
                    "object_to_first_vertex_mm","last_vertex_to_image_mm",
                    "stop_to_first_vertex_mm","last_vertex_to_exit_pupil_mm",
                )}}


def geometry_vector(candidate):
    entity = candidate["entity"]
    return [*entity["radii_mm"],*entity["center_thicknesses_mm"],entity["air_gap_mm"]]


def rounding_interval(value):
    value = float(value)
    center = Fraction.from_float(value)
    previous = nextafter(value, -float("inf"))
    following = nextafter(value, float("inf"))
    low = ((Fraction.from_float(previous)+center)/2 if isfinite(previous)
           else center-(Fraction.from_float(following)-center)/2)
    high = ((center+Fraction.from_float(following))/2 if isfinite(following)
            else center+(center-Fraction.from_float(previous))/2)
    return low, high


def precision_overlap(left, right):
    """Closed storage-rounding intervals, including ties at shared boundaries."""
    return all(max(a[0],b[0]) <= min(a[1],b[1]) for a,b in zip(left,right))


def relative_distance(left, right):
    return max(abs(a-b)/max(abs(a),abs(b)) if a or b else 0.0
               for a,b in zip(left,right))


class Registry:
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prescriptions (
                sample_sha256 TEXT PRIMARY KEY,
                canonical_sample TEXT NOT NULL,
                validation_status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS geometry_variants (
                sample_sha256 TEXT NOT NULL REFERENCES prescriptions,
                geometry_sha256 TEXT NOT NULL,
                full_sha256 TEXT NOT NULL,
                canonical_geometry TEXT NOT NULL,
                canonical_full TEXT NOT NULL,
                PRIMARY KEY(sample_sha256, geometry_sha256, full_sha256)
            );
            CREATE TABLE IF NOT EXISTS occurrences (
                source_path TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                record_key TEXT NOT NULL,
                case_id INTEGER,
                sampling_attempt_id INTEGER,
                sample_sha256 TEXT NOT NULL REFERENCES prescriptions,
                PRIMARY KEY(source_sha256,record_key)
            );
            CREATE TABLE IF NOT EXISTS batch_cases (
                batch_path TEXT NOT NULL,
                case_id INTEGER NOT NULL,
                sample_sha256 TEXT NOT NULL REFERENCES prescriptions,
                PRIMARY KEY(batch_path,case_id)
            );
            CREATE TABLE IF NOT EXISTS execution_refs (
                batch_path TEXT NOT NULL,
                case_id INTEGER NOT NULL,
                rule INTEGER NOT NULL,
                result_path TEXT NOT NULL,
                result_sha256 TEXT NOT NULL,
                sample_sha256 TEXT NOT NULL REFERENCES prescriptions,
                PRIMARY KEY(batch_path,case_id,rule)
            );
        """)
        definitions = {
            "sampling_schema":"two-lens-sampling-binary64-v1",
            "canonical_numeric_format":"finite binary64 float.hex(); signed zero normalized",
            "length_unit":"mm",
            "fixed_parameters":FIXED,
        }
        for key,value in definitions.items():
            payload = json.dumps(value,sort_keys=True)
            existing = self.connection.execute("SELECT value_json FROM metadata WHERE key=?",(key,)).fetchone()
            if existing and existing[0] != payload:
                raise ValueError("Registry schema or fixed parameters differ")
            self.connection.execute("INSERT OR IGNORE INTO metadata VALUES (?,?)",(key,payload))

    def register(self, candidate, source_path, source_sha256, record_key):
        digest,payload = sample_fingerprint(candidate)
        status = "valid" if candidate["passed"] else "rejected"
        existing = self.connection.execute(
            "SELECT canonical_sample,validation_status FROM prescriptions WHERE sample_sha256=?",
            (digest,)).fetchone()
        if existing and existing != (payload,status):
            raise ValueError("Inconsistent validation or fingerprint collision")
        self.connection.execute("INSERT OR IGNORE INTO prescriptions VALUES (?,?,?)",
                                (digest,payload,status))
        if candidate["passed"]:
            geometry,canonical_geometry = fingerprint(geometry_parameters(candidate),"two-lens-geometry-v1")
            full,canonical_full = fingerprint(full_parameters(candidate),"two-lens-full-v1")
            self.connection.execute("INSERT OR IGNORE INTO geometry_variants VALUES (?,?,?,?,?)",
                                    (digest,geometry,full,canonical_geometry,canonical_full))
        self.connection.execute("INSERT OR IGNORE INTO occurrences VALUES (?,?,?,?,?,?)",(
            source_path,source_sha256,record_key,candidate.get("id"),
            candidate.get("sampling_attempt_id",candidate.get("id")),digest))
        return digest

    def lookup(self, candidate):
        digest,_ = sample_fingerprint(candidate)
        row = self.connection.execute(
            "SELECT validation_status FROM prescriptions WHERE sample_sha256=?",(digest,)).fetchone()
        return {"sample_sha256":digest, "registered":row is not None,
                "validation_status":row[0] if row else None}

    def lookup_geometry(self, candidate):
        digest,_ = fingerprint(geometry_parameters(candidate),"two-lens-geometry-v1")
        rows = self.connection.execute(
            "SELECT DISTINCT sample_sha256 FROM geometry_variants WHERE geometry_sha256=?",
            (digest,)).fetchall()
        return {"geometry_sha256":digest, "registered":bool(rows),
                "matching_sampling_fingerprints":[row[0] for row in rows]}

    def close(self):
        self.connection.commit()
        self.connection.close()
