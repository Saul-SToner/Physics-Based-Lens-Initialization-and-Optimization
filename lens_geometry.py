"""Geometry checks for centered, coaxial spherical elements in air."""

from dataclasses import dataclass
from math import copysign, isfinite, isinf, sqrt

from two_lens import TwoLensDesign


@dataclass(frozen=True)
class GeometryResult:
    edge_thickness_mm: tuple[float, float]
    axial_air_gap_mm: float
    minimum_air_gap_mm: float
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations


def _sag(radius: float, height: float) -> float:
    if isinf(radius):
        return 0.0
    ratio = height / abs(radius)
    if ratio > 1:
        raise ValueError("Clear semi-aperture exceeds the magnitude of a surface radius")
    return copysign(height * ratio / (1 + sqrt(1 - ratio * ratio)), radius)


def check_lens_geometry(
    design: TwoLensDesign,
    center_thickness_mm: tuple[float, float],
    semi_aperture_mm: tuple[float, float],
    *,
    minimum_clearance_mm: float = 1e-9,
) -> GeometryResult:
    """Place each thin-lens reference plane halfway between its vertices.

    Both surfaces of each element use the same clear semi-aperture. Checks
    apply over the clear apertures and use signed radii in propagation order.
    """
    t1, t2 = center_thickness_mm
    h1, h2 = semi_aperture_mm
    if not all(isfinite(value) and value > 0 for value in (design.d12, t1, t2, h1, h2)):
        raise ValueError("Spacing, center thicknesses and semi-apertures must be finite and positive")
    if not all(map(isfinite, (design.k1, design.k2))):
        raise ValueError("Lens powers must be finite")
    if not isfinite(minimum_clearance_mm) or minimum_clearance_mm < 0:
        raise ValueError("Minimum clearance must be finite and nonnegative")

    r1, r2, r3, r4 = design.radii_mm
    edge1 = t1 + _sag(r2, h1) - _sag(r1, h1)
    edge2 = t2 + _sag(r4, h2) - _sag(r3, h2)
    axial_gap = design.d12 - (t1 / 2 + t2 / 2)
    common_height = min(h1, h2)
    # For coaxial spherical surfaces the sag difference is monotonic with
    # radial height, so its minimum occurs on-axis or at the common edge.
    edge_gap = axial_gap + _sag(r3, common_height) - _sag(r2, common_height)
    minimum_gap = min(axial_gap, edge_gap)
    violations = tuple(reason for value, reason in (
        (min(t1, edge1), "lens_1_insufficient_thickness"),
        (min(t2, edge2), "lens_2_insufficient_thickness"),
        (axial_gap, "insufficient_axial_air_gap"),
        (edge_gap, "insufficient_air_gap_at_common_edge"),
    ) if not isfinite(value) or value <= minimum_clearance_mm)
    return GeometryResult((edge1, edge2), axial_gap, minimum_gap, violations)
