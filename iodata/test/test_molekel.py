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
"""Test iodata.molekel module."""

from numpy.testing import assert_equal, assert_allclose

from . common import check_orthonormal
from .. iodata import load_one
from .. utils import angstrom, shells_to_nbasis
from .. overlap import compute_overlap

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_mkl_ethanol():
    with path('iodata.test.data', 'ethanol.mkl') as fn_mkl:
        mol = load_one(str(fn_mkl))

    # Direct checks with mkl file
    assert_equal(mol.numbers.shape, (9,))
    assert_equal(mol.numbers[0], 1)
    assert_equal(mol.numbers[4], 6)
    assert_equal(mol.coordinates.shape, (9, 3))
    assert_allclose(mol.coordinates[2, 1] / angstrom, 2.239037, atol=1.e-5)
    assert_allclose(mol.coordinates[5, 2] / angstrom, 0.948420, atol=1.e-5)
    assert_equal(shells_to_nbasis(mol.obasis["shell_types"]), 39)
    assert_allclose(mol.obasis['alphas'][0], 18.731137000)
    assert_allclose(mol.obasis['alphas'][10], 7.868272400)
    assert_allclose(mol.obasis['alphas'][-3], 2.825393700)
    # assert mol.obasis.con_coeffs[5] == 0.989450608
    # assert mol.obasis.con_coeffs[7] == 2.079187061
    # assert mol.obasis.con_coeffs[-1] == 0.181380684
    assert_equal(mol.obasis["shell_map"][:5], [0, 0, 1, 1, 1])
    assert_equal(mol.obasis["shell_types"][:5], [0, 0, 0, 0, 1])
    assert_equal(mol.obasis['nprims'][-5:], [3, 1, 1, 3, 1])
    assert_equal(mol.orb_alpha_coeffs.shape, (39, 39))
    assert_equal(mol.orb_alpha_energies.shape, (39,))
    assert_equal(mol.orb_alpha_occs.shape, (39,))
    assert_equal(mol.orb_alpha_occs[:13], 1.0)
    assert_equal(mol.orb_alpha_occs[13:], 0.0)
    assert_allclose(mol.orb_alpha_energies[4], -1.0206976)
    assert_allclose(mol.orb_alpha_energies[-1], 2.0748685)
    assert_allclose(mol.orb_alpha_coeffs[0, 0], 0.0000119)
    assert_allclose(mol.orb_alpha_coeffs[1, 0], -0.0003216)
    assert_allclose(mol.orb_alpha_coeffs[-1, -1], -0.1424743)


def test_load_mkl_li2():
    with path('iodata.test.data', 'li2.mkl') as fn_mkl:
        mol = load_one(str(fn_mkl))

    # Check normalization
    olp = compute_overlap(**mol.obasis)
    check_orthonormal(mol.orb_alpha_coeffs, olp, 1e-5)
    check_orthonormal(mol.orb_beta_coeffs, olp, 1e-5)


def test_load_mkl_h2():
    with path('iodata.test.data', 'h2_sto3g.mkl') as fn_mkl:
        mol = load_one(str(fn_mkl))
    olp = compute_overlap(**mol.obasis)
    check_orthonormal(mol.orb_alpha_coeffs, olp, 1e-5)
