from copy import deepcopy
from fractions import Fraction
from math import nextafter
from pathlib import Path
import tempfile
import unittest

from prescription_registry import Registry, fingerprint, precision_overlap, rounding_interval, sample_fingerprint


class PrescriptionRegistryTests(unittest.TestCase):
    def candidate(self):
        return {"inputs":[-2.0,1.0,-1.0,20.0,-2.0],"id":1,"passed":True,
                "entity":{"radii_mm":[2.0,-2.0,3.0,-3.0],"center_thicknesses_mm":[1.0,1.0],
                          "air_gap_mm":5.0,"index":1.5,"semi_apertures_mm":[1.0,1.0],
                          "object_to_first_vertex_mm":1.5,"last_vertex_to_image_mm":0.5,
                          "stop_to_first_vertex_mm":0.5,"last_vertex_to_exit_pupil_mm":19.5}}

    def test_lossless_numbers_and_signed_zero(self):
        self.assertEqual(fingerprint({"x":-0.0},"test"),fingerprint({"x":0.0},"test"))
        self.assertEqual(fingerprint({"x":1},"test"),fingerprint({"x":1.0},"test"))
        self.assertNotEqual(fingerprint({"x":1.0},"test"),
                            fingerprint({"x":nextafter(1.0,2.0)},"test"))

    def test_true_rounding_midpoints_at_power_of_two(self):
        low,high = rounding_interval(1.0)
        self.assertEqual(low,(Fraction.from_float(nextafter(1.0,0.0))+1)/2)
        self.assertEqual(high,(Fraction.from_float(nextafter(1.0,2.0))+1)/2)
        self.assertNotEqual(1-low,high-1)
        adjacent=nextafter(1.0,2.0)
        self.assertTrue(precision_overlap([rounding_interval(1.0)],[rounding_interval(adjacent)]))
        self.assertFalse(precision_overlap([rounding_interval(1.0)],
                         [rounding_interval(nextafter(adjacent,2.0))]))

    def test_ids_do_not_change_identity(self):
        a=self.candidate()
        b=deepcopy(a)
        b["id"]=999
        self.assertEqual(sample_fingerprint(a),sample_fingerprint(b))

    def test_registry_aliases_and_geometry_query(self):
        with tempfile.TemporaryDirectory() as folder:
            registry=Registry(Path(folder)/"registry.sqlite")
            try:
                row=self.candidate()
                registry.register(row,"source1","sha1","row1")
                registry.register(row,"source2","sha2","row1")
                self.assertEqual(registry.connection.execute("SELECT count(*) FROM prescriptions").fetchone()[0],1)
                self.assertTrue(registry.lookup({"inputs":row["inputs"]})["registered"])
                changed=deepcopy(row)
                changed["inputs"][0]=-3.0
                self.assertFalse(registry.lookup(changed)["registered"])
                self.assertTrue(registry.lookup_geometry(changed)["registered"])
            finally:
                registry.close()


if __name__=="__main__":
    unittest.main()
