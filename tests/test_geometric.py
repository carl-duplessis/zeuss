"""Small-grade Clifford algebra: real algebraic properties, verified directly."""
import numpy as np

from zeuss.tier2_substrate.geometric import Multivector, apply_rotor, geometric_product, rotor


def test_geometric_product_is_associative():
    rng = np.random.default_rng(0)
    n = 3
    a = Multivector(n, rng.normal(size=2**n))
    b = Multivector(n, rng.normal(size=2**n))
    c = Multivector(n, rng.normal(size=2**n))
    lhs = geometric_product(geometric_product(a, b), c).coeffs
    rhs = geometric_product(a, geometric_product(b, c)).coeffs
    assert np.allclose(lhs, rhs)


def test_known_cl2_blade_identities():
    # e1*e2 = e12 (+1); e2*e1 = e12 (-1); e12*e12 = -1 - the relation that
    # makes the even subalgebra behave like the imaginary unit.
    e1 = Multivector.basis_vector(2, 0)
    e2 = Multivector.basis_vector(2, 1)
    e12 = Multivector.basis_bivector(2, 0, 1)

    assert np.allclose(geometric_product(e1, e2).coeffs, e12.coeffs)
    assert np.allclose(geometric_product(e2, e1).coeffs, -e12.coeffs)
    assert np.allclose(geometric_product(e12, e12).coeffs, -Multivector.scalar(2, 1.0).coeffs)


def test_rotor_preserves_vector_magnitude():
    bivector = Multivector.basis_bivector(2, 0, 1)
    v = Multivector.basis_vector(2, 0)
    for theta in (0.3, 1.0, 2.7, -1.5):
        rotated = apply_rotor(rotor(bivector, theta), v)
        assert abs(rotated.norm() - v.norm()) < 1e-9


def test_rotor_composition_adds_angles_for_coplanar_rotations():
    bivector = Multivector.basis_bivector(2, 0, 1)
    composed = geometric_product(rotor(bivector, 0.4), rotor(bivector, 0.9))
    direct = rotor(bivector, 1.3)
    assert np.allclose(composed.coeffs, direct.coeffs)


def test_cl2_even_subalgebra_matches_complex_multiplication():
    """Cl(2,0)'s even subalgebra (scalar + pseudoscalar) is isomorphic to C -
    the genuine bridge back to hypervectors.py's complex-phasor algebra."""

    def to_mv(z: complex) -> Multivector:
        c = np.zeros(4)
        c[0] = z.real
        c[3] = z.imag
        return Multivector(2, c)

    def from_mv(mv: Multivector) -> complex:
        return complex(mv.coeffs[0], mv.coeffs[3])

    for z1, z2 in [(1 + 2j, 3 - 1j), (0.5 + 0.5j, 2 + 0j), (-1 + 1j, -1 - 1j)]:
        expected = z1 * z2
        actual = from_mv(geometric_product(to_mv(z1), to_mv(z2)))
        assert abs(actual - expected) < 1e-9


def test_max_grade_is_enforced():
    import pytest

    from zeuss.tier2_substrate.geometric import MAX_GRADE, _table

    with pytest.raises(ValueError):
        _table(MAX_GRADE + 1)
