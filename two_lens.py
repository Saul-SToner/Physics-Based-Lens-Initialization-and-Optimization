"""Validated paraxial predesign of two thin lenses in air."""

from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real


ABS_LENGTH_MM = 1e-9
ABS_POWER_PER_MM = 1e-12
ABS_MAGNIFICATION = 1e-12
REL_TOL = 1e-9


class InvalidStructure(ValueError):
    """A rejected predesign, with its original input and machine-readable reason."""

    def __init__(self, reason: str, inputs: tuple):
        self.reason = reason
        self.inputs = inputs
        super().__init__(f"{reason}: {inputs!r}")


@dataclass(frozen=True)
class TwoLensDesign:
    d12: float
    k1: float
    k2: float
    efl: float

    @property
    def radii_mm(self) -> tuple[float, float, float, float]:
        """Front/back radii of lens 1, then lens 2, for symmetric n=1.5 lenses."""
        r1 = 1 / self.k1 if self.k1 else float("inf")
        r2 = 1 / self.k2 if self.k2 else float("inf")
        return r1, -r1, r2, -r2


def design_two_lens(
    l1: float, l2_prime: float, pupil_l1: float, pupil_l2_prime: float,
    magnification: float,
) -> TwoLensDesign:
    """Use signed axial coordinates relative to the adjacent lens, in mm.

    Positive coordinates point along propagation. The magnification is signed.
    Return spacing and EFL in mm and lens powers in inverse mm.
    """
    inputs = (l1, l2_prime, pupil_l1, pupil_l2_prime, magnification)
    try:
        finite = all(
            isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)
            for value in inputs
        )
    except OverflowError:
        finite = False
    if not finite:
        raise InvalidStructure("nonfinite_or_nonreal_input", inputs)

    l1, l2_prime, pupil_l1, pupil_l2_prime, m = map(float, inputs)
    if abs(m) <= ABS_MAGNIFICATION:
        raise InvalidStructure("zero_or_near_zero_magnification", inputs)

    # The object-to-image ABCD matrix has A=M, B=0 and D=1/M.
    # Requiring B=0 between the pupil planes determines its remaining entry C.
    x = l1 - pupil_l1
    y = pupil_l2_prime - l2_prime
    if not isfinite(x) or not isfinite(y):
        raise InvalidStructure("nonfinite_solution", inputs)
    if isclose(l1, pupil_l1, rel_tol=REL_TOL, abs_tol=ABS_LENGTH_MM) or isclose(
        l2_prime, pupil_l2_prime, rel_tol=REL_TOL, abs_tol=ABS_LENGTH_MM
    ):
        raise InvalidStructure("degenerate_pupil_conjugates", inputs)
    c = -m / y - (1 / m) / x

    # Undo the translations from the object and to the image. The resulting
    # lens-to-lens matrix is L(K2) P(d12) L(K1), whose B entry is d12.
    a = -l1
    b = l2_prime
    lens_a = m - b * c
    lens_d = 1 / m - a * c
    d12 = -a * m - b * lens_d
    if not all(map(isfinite, (lens_a, lens_d, d12))):
        raise InvalidStructure("nonfinite_solution", inputs)
    if d12 <= ABS_LENGTH_MM:
        raise InvalidStructure("nonpositive_or_negligible_spacing", inputs)

    k1 = (1 - lens_a) / d12
    k2 = (1 - lens_d) / d12
    coupling = d12 * k1 * k2
    power = k1 + k2 - coupling
    if not all(map(isfinite, (k1, k2, coupling, power))):
        raise InvalidStructure("nonfinite_solution", inputs)
    if abs(power) <= max(ABS_POWER_PER_MM, REL_TOL * max(abs(k1), abs(k2), abs(coupling))):
        raise InvalidStructure("afocal_or_near_afocal", inputs)

    # Trace the two basis rays through the solved lenses and translations.
    for start, end, expected_m, error in (
        (l1, l2_prime, m, "object_image_residual"),
        (pupil_l1, pupil_l2_prime, None, "pupil_conjugate_residual"),
    ):
        height_at_second = -start + d12 * (1 + k1 * start)
        end_shift = end * (1 + k1 * start - k2 * height_at_second)
        if not isfinite(height_at_second) or not isclose(
            height_at_second, -end_shift, rel_tol=REL_TOL, abs_tol=ABS_LENGTH_MM
        ):
            raise InvalidStructure(error, inputs)
        if expected_m is not None:
            image_height = (1 - d12 * k1) - end * power
            if not isclose(image_height, expected_m, rel_tol=REL_TOL, abs_tol=ABS_MAGNIFICATION):
                raise InvalidStructure("magnification_residual", inputs)

    return TwoLensDesign(d12, k1, k2, 1 / power)
