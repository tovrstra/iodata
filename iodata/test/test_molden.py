# -*- coding: utf-8 -*-
# IODATA is an input and output module for quantum chemistry.
#
# Copyright (C) 2011-2019 The IODATA Development Team
#
# This file is part of IODATA.
#
# IODATA is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# IODATA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
# pragma pylint: disable=invalid-name,no-member
"""Test iodata.molden module."""

import os
import numpy as np
from numpy.testing import assert_allclose

from .common import compute_mulliken_charges, compare_mols, check_orthonormal
from ..iodata import load_one, dump_one
from ..overlap import compute_overlap

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_molden_li2_orca():
    with path('iodata.test.data', 'li2.molden.input') as fn_molden:
        mol = load_one(str(fn_molden))

    # Checkt title
    assert mol.title == 'Molden file created by orca_2mkl for BaseName=li2'

    # Check normalization
    olp = compute_overlap(**mol.obasis)
    check_orthonormal(mol.orb_alpha_coeffs, olp, 1e-5)
    check_orthonormal(mol.orb_beta_coeffs, olp, 1e-5)

    # Check Mulliken charges
    charges = compute_mulliken_charges(mol)
    expected_charges = np.array([0.5, 0.5])
    assert_allclose(charges, expected_charges, atol=1.e-5)


def test_load_molden_h2o_orca():
    with path('iodata.test.data', 'h2o.molden.input') as fn_molden:
        mol = load_one(str(fn_molden))

    # Checkt title
    assert mol.title == 'Molden file created by orca_2mkl for BaseName=h2o'

    # Check normalization
    olp = compute_overlap(**mol.obasis)
    check_orthonormal(mol.orb_alpha_coeffs, olp, 1e-5)

    # Check Mulliken charges
    charges = compute_mulliken_charges(mol)
    expected_charges = np.array([-0.816308, 0.408154, 0.408154])
    assert_allclose(charges, expected_charges, atol=1.e-5)


def test_load_molden_nh3_molden_pure():
    # The file tested here is created with molden. It should be read in
    # properly without altering normalization and sign conventions.
    with path('iodata.test.data', 'nh3_molden_pure.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert_allclose(charges, molden_charges, atol=1.e-3)


def test_load_molden_nh3_molden_cart():
    # The file tested here is created with molden. It should be read in
    # properly without altering normalization and sign conventions.
    with path('iodata.test.data', 'nh3_molden_cart.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.3138, -0.4300, -0.0667, 0.1829])
    assert_allclose(charges, molden_charges, atol=1.e-3)


def test_load_molden_nh3_orca():
    # The file tested here is created with ORCA. It should be read in
    # properly by altering normalization and sign conventions.
    with path('iodata.test.data', 'nh3_orca.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert_allclose(charges, molden_charges, atol=1.e-3)


def test_load_molden_nh3_psi4():
    # The file tested here is created with PSI4 (pre 1.0). It should be read in
    # properly by altering normalization conventions.
    with path('iodata.test.data', 'nh3_psi4_1.0.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert_allclose(charges, molden_charges, atol=1.e-3)


def test_load_molden_nh3_psi4_1():
    # The file tested here is created with PSI4 (version 1.0).
    # It should be read in properly by renormalizing the contractions.
    with path('iodata.test.data', 'nh3_psi4_1.0.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert_allclose(charges, molden_charges, atol=1.e-3)


def test_load_molden_he2_ghost_psi4_1():
    # The file tested here is created with PSI4 (version 1.0).
    # It should be read in properly by ignoring the ghost atoms.
    with path('iodata.test.data', 'he2_ghost_psi4_1.0.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    np.testing.assert_equal(mol.pseudo_numbers, np.array([2.0]))
    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol, np.array([0.0, 2.0]))
    molden_charges = np.array([-0.0041, 0.0041])
    assert_allclose(charges, molden_charges, atol=5e-4)


def test_load_molden_nh3_molpro2012():
    # The file tested here is created with MOLPRO2012.
    with path('iodata.test.data', 'nh3_molpro2012.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check Mulliken charges.
    # Comparison with numbers from the Molden program output.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.0381, -0.2742, 0.0121, 0.2242])
    assert_allclose(charges, molden_charges, atol=1.e-3)


def test_load_molden_neon_turbomole():
    # The file tested here is created with Turbomole 7.1.
    with path('iodata.test.data',
              'neon_turbomole_def2-qzvp.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check Mulliken charges.
    charges = compute_mulliken_charges(mol)
    assert abs(charges).max() < 1e-3


def test_load_molden_nh3_turbomole():
    # The file tested here is created with Turbomole 7.1
    with path('iodata.test.data', 'nh3_turbomole.molden') as fn_molden:
        mol = load_one(str(fn_molden))
    # Check Mulliken charges.
    # Comparison with numbers from the Turbomole output.
    # These are slightly different than in the other tests because we are using
    # Cartesian functions.
    charges = compute_mulliken_charges(mol)
    molden_charges = np.array([0.03801, -0.27428, 0.01206, 0.22421])
    assert_allclose(charges, molden_charges, atol=1.e-3)


def check_load_dump_consistency(fn, tmpdir):
    """Check if data is preserved after dumping and loading a Molden file.

    Parameters
    ----------
    fn : str
        The Molden filename to load
    tmpdir : str
        The temporary directory to dump and load the file.

    """
    with path('iodata.test.data', fn) as file_name:
        mol1 = load_one(str(file_name))
    fn_tmp = os.path.join(tmpdir, 'foo.molden')
    dump_one(mol1, fn_tmp)
    mol2 = load_one(fn_tmp)
    compare_mols(mol1, mol2)


def test_load_dump_consistency_h2o(tmpdir):
    check_load_dump_consistency('h2o.molden.input', tmpdir)


def test_load_dump_consistency_li2(tmpdir):
    check_load_dump_consistency('li2.molden.input', tmpdir)


def test_load_dump_consistency_f(tmpdir):
    check_load_dump_consistency('F.molden', tmpdir)


def test_load_dump_consistency_nh3_molden_pure(tmpdir):
    check_load_dump_consistency('nh3_molden_pure.molden', tmpdir)


def test_load_dump_consistency_nh3_molden_cart(tmpdir):
    check_load_dump_consistency('nh3_molden_cart.molden', tmpdir)
