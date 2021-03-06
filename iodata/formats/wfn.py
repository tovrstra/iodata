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
# pragma pylint: disable=wrong-import-order,invalid-name
"""Module for handling GUASSIAN/GAMESS-US WNF file format."""


import numpy as np

from typing import Tuple, List, TextIO, Dict

from ..overlap import init_scales
from ..periodic import sym2num
from ..utils import MolecularOrbitals, LineIterator


__all__ = ['load_wfn_low', 'get_permutation_orbital',
           'get_permutation_basis', 'get_mask', 'load']


patterns = ['*.wfn']


def _load_helper_num(lit: LineIterator) -> List[int]:
    """Read number of orbitals, primitives and atoms."""
    line = next(lit)
    if not line.startswith('GAUSSIAN'):
        lit.error("Expecting line to start with GAUSSIAN.")
    return [int(i) for i in line.split() if i.isdigit()]


def _load_helper_coordinates(lit: LineIterator, num_atoms: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read the coordiantes of the atoms."""
    numbers = np.empty(num_atoms, int)
    coordinates = np.empty((num_atoms, 3), float)
    for atom in range(num_atoms):
        words = next(lit).split()
        numbers[atom] = sym2num[words[0].title()]
        coordinates[atom, :] = [words[4], words[5], words[6]]
    return numbers, coordinates


def _load_helper_section(lit: LineIterator, num_primitives: int, start: str, skip: int) -> List:
    """Read CENTRE ASSIGNMENTS, TYPE ASSIGNMENTS, and EXPONENTS sections."""
    section = []
    while len(section) < num_primitives:
        line = next(lit)
        assert line.startswith(start)
        words = line.split()
        section.extend(words[skip:])
    assert len(section) == num_primitives
    return section


def _load_helper_mo(lit: LineIterator, num_primitives: int) -> Tuple[str, str, str, List[str]]:
    """Read one section of MO information."""
    line = next(lit)
    assert line.startswith('MO')
    words = line.split()
    count = words[1]
    occ, energy = words[-5], words[-1]
    coeffs = _load_helper_section(lit, num_primitives, ' ', 0)
    coeffs = [i.replace('D', 'E') for i in coeffs]
    return count, occ, energy, coeffs


def _load_helper_energy(lit: LineIterator) -> float:
    """Read energy."""
    line = next(lit).lower()
    while 'energy' not in line and line is not None:
        line = next(lit).lower()
    energy = float(line.split('energy =')[1].split()[0])
    return energy


def load_wfn_low(lit: LineIterator) -> Tuple:
    """Load data from a WFN file into arrays.

    Parameters
    ----------
    lit
        The line iterator to read the data from.
    """
    # read sections of wfn file
    title = next(lit).strip()
    num_mo, num_primitives, num_atoms = _load_helper_num(lit)
    numbers, coordinates = _load_helper_coordinates(lit, num_atoms)
    # centers are indexed from zero in HORTON
    centers = np.array([
        int(i) - 1 for i in
        _load_helper_section(lit, num_primitives, 'CENTRE ASSIGNMENTS', 2)])
    type_assignment = np.array([
        int(i) for i in
        _load_helper_section(lit, num_primitives, 'TYPE ASSIGNMENTS', 2)])
    exponent = np.array([
        float(i.replace('D', 'E')) for i in
        _load_helper_section(lit, num_primitives, 'EXPONENTS', 1)])
    mo_count = np.empty(num_mo, int)
    mo_occ = np.empty(num_mo, float)
    mo_energy = np.empty(num_mo, float)
    coefficients = np.empty([num_primitives, num_mo], float)
    for mo in range(num_mo):
        mo_count[mo], mo_occ[mo], mo_energy[mo], coefficients[:, mo] = \
            _load_helper_mo(lit, num_primitives)
    energy = _load_helper_energy(lit)
    return title, numbers, coordinates, centers, type_assignment, exponent, \
           mo_count, mo_occ, mo_energy, coefficients, energy


def get_permutation_orbital(type_assignment: np.ndarray) -> np.ndarray:
    """Permute each type of orbital to get the proper order for HORTON."""
    num_primitive = len(type_assignment)
    permutation = np.arange(num_primitive)
    # degeneracy of {s:1, p:3, d:6, f:10, g:15, h:21}
    degeneracy = {1: 1, 2: 3, 5: 6, 11: 10, 23: 15, 36: 21}
    index = 0
    while index < num_primitive:
        value = type_assignment[index]
        length = degeneracy[value]
        if value != 1 and value == type_assignment[index + 1]:
            sub_count = 1
            while index + sub_count < num_primitive and type_assignment[index + sub_count] == value:
                sub_count += 1
            sub_type = np.empty(sub_count, int)
            sub_type[:] = permutation[index: index + sub_count]
            for i in range(sub_count):
                permutation[index: index + length] = sub_type[i] + np.arange(length) * sub_count
                index += length
        else:
            index += length
    assert (np.sort(permutation) == np.arange(num_primitive)).all()
    return permutation


def get_permutation_basis(type_assignment: np.ndarray) -> np.ndarray:
    """
    Permute the basis functions to get the proper order for HORTON.

    Permutation conventions are as follows:

     d orbitals:
       wfn:     [5, 6, 7, 8, 9, 10]
       HORTON:  [5, 8, 9, 6, 10, 7]
       permute: [0, 3, 4, 1, 5, 2]

     f orbitals:
       wfn:     [11, 12, 13, 17, 14, 15, 18, 19, 16, 20]
       HORTON:  [11, 14, 15, 17, 20, 18, 12, 16, 19, 13]
       permute: [0, 4, 5, 3, 9, 6, 1, 8, 7, 2]

     g orbital:
       wfn:     [23, 29, 32, 27, 22, 28, 35, 34, 26, 31, 33, 30, 25, 24, 21]
       HORTON:  [21, 24, 25, 30, 33, 31, 26, 34, 35, 28, 22, 27, 32, 29, 23]
       permute: [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

     h orbital:
       wfn:     [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]
       HORTON:  [56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36]
       permute: [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    """
    permutation = get_permutation_orbital(type_assignment)
    type_assignment = type_assignment[permutation]
    for count, value in enumerate(type_assignment):
        if value == 5:
            # d-orbitals
            permute = [0, 3, 4, 1, 5, 2]
            permutation[count: count + 6] = permutation[count: count + 6][permute]
        elif value == 11:
            # f-orbitals
            permute = [0, 4, 5, 3, 9, 6, 1, 8, 7, 2]
            permutation[count: count + 10] = permutation[count: count + 10][permute]
        elif value == 23:
            # g-orbitals
            permute = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
            permutation[count: count + 15] = permutation[count:count + 15][permute]
        elif value == 36:
            # h-orbitals
            permutation[count: count + 21] = permutation[count: count + 21][::-1]
    return permutation


def get_mask(type_assignment: np.ndarray) -> np.ndarray:
    """Return array to mask orbital types other than s, 1st_p, 1st_d, 1st_f, 1st_g, 1st_h."""
    # index of [s, 1st_p, 1st_d, 1st_f, 1st_g, 1st_h]
    temp = [1, 2, 5, 11, 21, 36]
    mask = np.array([i in temp for i in type_assignment])
    return mask


def load(lit: LineIterator) -> Dict:
    """Load data from a WFN file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out : dict
        Output dictionary containing ``title``, ``coordinates``, ``numbers``, ``energy``,
        ``obasis`` & ``orb_alpha`` keys and their corresponding values. It may contain
        ``orb_beta`` key and its value as well.

    """
    (title, numbers, coordinates, centers, type_assignment, exponents,
     mo_count, mo_occ, mo_energy, coefficients, energy) = load_wfn_low(lit)
    permutation = get_permutation_basis(type_assignment)
    # permute arrays containing wfn data
    type_assignment = type_assignment[permutation]
    mask = get_mask(type_assignment)
    reduced_size = np.array(mask, int).sum()
    num = coefficients.shape[1]
    alphas = np.empty(reduced_size)
    alphas[:] = exponents[permutation][mask]
    assert (centers == centers[permutation]).all()
    shell_map = centers[mask]
    # cartesian basis: {S:0, P:1, D:2, F:3, G:4, H:5}
    shell = {1: 0, 2: 1, 5: 2, 11: 3, 21: 4, 36: 5}
    shell_types = type_assignment[mask]
    shell_types = np.array([shell[i] for i in shell_types])
    assert shell_map.size == shell_types.size == reduced_size
    nprims = np.ones(reduced_size, int)
    con_coeffs = np.ones(reduced_size)
    # build basis set
    obasis = {"centers": coordinates, "shell_map": shell_map, "nprims": nprims,
              "shell_types": shell_types, "alphas": alphas, "con_coeffs": con_coeffs}
    coefficients = coefficients[permutation]
    scales, dummy = init_scales(obasis["alphas"], obasis["nprims"], obasis["shell_types"])
    coefficients /= scales.reshape(-1, 1)
    # make the wavefunction
    if mo_occ.max() > 1.0:
        # close shell system
        mo_type = 'restricted'
        na_orb = len(mo_occ)
        nb_orb = len(mo_occ)
    else:
        # open shell system
        mo_type = 'unrestricted'
        # counting the number of alpha and beta orbitals
        n = 1
        while n < num and mo_energy[n] >= mo_energy[n - 1] and mo_count[n] == mo_count[n - 1] + 1:
            n += 1
        na_orb = n
        nb_orb = len(mo_occ) - n
    # create a MO namedtuple
    mo = MolecularOrbitals(mo_type, na_orb, nb_orb, mo_occ, coefficients, None, mo_energy)

    result = {
        'title': title,
        'coordinates': coordinates,
        'numbers': numbers,
        'obasis': obasis,
        'mo': mo,
        'energy': energy,
    }
    return result
