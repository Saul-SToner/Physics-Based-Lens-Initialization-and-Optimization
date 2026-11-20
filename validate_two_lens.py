"""Independent paraxial cross-check using DCC-Lab RayTracing."""

from dataclasses import dataclass
from math import isfinite

from raytracing import Lens, Space


@dataclass(frozen=True)
class Residual:
    value: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return isfinite(self.value) and abs(self.value) <= self.tolerance


@dataclass(frozen=True)
class ValidationResult:
    object_image_b_mm: Residual
    magnification_error: Residual
    pupil_b_mm: Residual
    efl_error_mm: Residual

    @property
    def passed(self) -> bool:
        return all(check.passed for check in (
            self.object_image_b_mm, self.magnification_error,
            self.pupil_b_mm, self.efl_error_mm,
        ))


def validate_two_lens(
    inputs: tuple[float, float, float, float, float],
    candidate: tuple[float, float, float, float],
) -> ValidationResult:
    """Validate signed axial inputs and a (d12, k1, k2, efl) candidate.

    Distances are in mm and powers are in inverse mm. Signed translations
    to virtual conjugates are intentional. Invalid numbers or spacing raise
    ValueError. Finite candidates that fail a residual yield a failed result.
    """
    l1, l2_prime, pupil_l1, pupil_l2_prime, magnification = inputs
    spacing, k1, k2, efl = candidate
    if not all(map(isfinite, (*inputs, *candidate))) or spacing <= 0:
        raise ValueError("Expected finite inputs and a positive lens spacing")

    first = Lens(f=1 / k1 if k1 else float("inf"))
    second = Lens(f=1 / k2 if k2 else float("inf"))
    system = second * Space(d=spacing) * first
    image = Space(d=l2_prime) * system * Space(d=-l1)
    pupil = Space(d=pupil_l2_prime) * system * Space(d=-pupil_l1)
    computed_efl = system.effectiveFocalLengths().f2

    relative_tolerance = 1e-9
    length_floor_mm = 1e-9
    magnification_floor = 1e-12
    return ValidationResult(
        object_image_b_mm=Residual(
            image.B, max(length_floor_mm, relative_tolerance * max(
                abs(l1), abs(l2_prime), spacing,
            )),
        ),
        magnification_error=Residual(
            image.A - magnification,
            max(magnification_floor, relative_tolerance * abs(magnification)),
        ),
        pupil_b_mm=Residual(
            pupil.B, max(length_floor_mm, relative_tolerance * max(
                abs(pupil_l1), abs(pupil_l2_prime), spacing,
            )),
        ),
        efl_error_mm=Residual(
            computed_efl - efl,
            max(length_floor_mm, relative_tolerance * abs(efl)),
        ),
    )
