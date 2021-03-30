# IODATA is an input and output module for quantum chemistry.
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
# --
"""Unit tests for the iodata.formats.mdf module."""


import os

from numpy.testing import assert_equal, assert_allclose
import pytest

from ..api import load_one, dump_one

try:
    from importlib_resources import path
except ImportError:
    from importlib.resources import path


def test_load_ethane():
    with path('iodata.test.data', 'ethane-oplsaa.mdf') as fn_mdf:
        mol = load_one(str(fn_mdf))
    assert mol.natom == 8
    assert mol.title == "ethane"
    assert_equal(mol.atnums, [6, 6, 1, 1, 1, 1, 1, 1])
    assert_equal(mol.bonds[:, :2], [[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [1, 7]])
    assert_equal(mol.atffparams["restypes"], "XXXX")
    assert_equal(mol.atffparams["resnums"], 1)
    assert_equal(mol.atffparams["attypes"], ["CT", "CT", "HC", "HC", "HC", "HC", "HC", "HC"])
    assert_allclose(mol.atffparams["charges"], [-0.180] * 2 + [0.060] * 6)
    assert_equal(mol.extra["atnames"], ["C1", "C2", "H3", "H4", "H5", "H6", "H7", "H8"])
    assert_equal(mol.extra["occupancies"], 1.0)
    assert_equal(mol.extra["bfactors"], 0.0)


@pytest.mark.parametrize("fn_base", ["ethane-oplsaa.mdf", "decane-oplsaa.mdf"])
def test_load_dump_consistency(tmpdir, fn_base):
    with path('iodata.test.data', fn_base) as fn_mdf:
        mol0 = load_one(str(fn_mdf))
    # write MDF file in a temporary folder & then read it
    fn_tmp = os.path.join(tmpdir, 'test.mdf')
    dump_one(mol0, fn_tmp)
    mol1 = load_one(fn_tmp)
    # Compare the two objectis
    assert mol0.title == mol1.title
    assert_equal(mol0.atnums, mol1.atnums)
    assert_equal(mol0.atffparams['attypes'], mol1.atffparams['attypes'])
    assert_equal(mol0.atffparams['charges'], mol1.atffparams['charges'])
    assert_equal(mol0.atffparams['restypes'], mol1.atffparams['restypes'])
    assert_equal(mol0.atffparams['resnums'], mol1.atffparams['resnums'])
    assert_equal(mol0.extra['atnames'], mol1.extra['atnames'])
    assert_equal(mol0.extra['occupancies'], mol1.extra['occupancies'])
    assert_equal(mol0.extra['bfactors'], mol1.extra['bfactors'])


def test_dump_from_xyz(tmpdir):
    with path('iodata.test.data', "water.xyz") as fn_xyz:
        mol0 = load_one(str(fn_xyz))
    # write MDF file in a temporary folder & then read it
    fn_tmp = os.path.join(tmpdir, 'water.mdf')
    dump_one(mol0, fn_tmp)
    mol1 = load_one(fn_tmp)
    assert mol0.title == mol1.title
    assert_equal(mol0.atnums, mol1.atnums)
    assert_equal(mol1.atffparams["attypes"], ["H", "O", "H"])
    assert_equal(mol1.atffparams["charges"], 0.0)
    assert_equal(mol1.atffparams["restypes"], "XXXX")
    assert_equal(mol1.atffparams["resnums"], 1)
    assert_equal(mol1.extra["atnames"], ["H1", "O2", "H3"])
