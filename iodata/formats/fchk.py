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
# pragma pylint: disable=wrong-import-order,invalid-name,too-many-statements,too-many-branches
"""Module for handling GAUSSIAN FCHK file format."""


import numpy as np

from typing import Dict, List, Tuple

from ..utils import LineIterator


__all__ = ['FCHKFile', 'load']


patterns = ['*.fchk']


def load(lit: LineIterator) -> Dict:
    """Load data from a GAUSSIAN FCHK file format.

    Parameters
    ----------
    lit
        The line iterator to read the data from.

    Returns
    -------
    out : dict
        Output dictionary containing ``title``, ``coordinates``, ``numbers``, ``pseudo_numbers``,
        ``obasis``, ``orb_alpha``, ``permutation``, ``energy`` & ``mulliken_charges`` keys and
        corresponding values. It may also contain ``npa_charges``, ``esp_charges``, ``orb_beta``,
        ``dm_full_mp2``, ``dm_spin_mp2``, ``dm_full_mp3``, ``dm_spin_mp3``, ``dm_full_cc``,
        ``dm_spin_cc``, ``dm_full_ci``, ``dm_spin_ci``, ``dm_full_scf``, ``dm_spin_scf``,
        ``polar``, ``dipole_moment`` & ``quadrupole_moment`` keys and their values as well.

    """
    fchk = _load_fchk_low(lit, [
        "Number of electrons", "Number of basis functions",
        "Number of independant functions",
        "Number of independent functions",
        "Number of alpha electrons", "Number of beta electrons",
        "Atomic numbers", "Current cartesian coordinates",
        "Shell types", "Shell to atom map", "Shell to atom map",
        "Number of primitives per shell", "Primitive exponents",
        "Contraction coefficients", "P(S=P) Contraction coefficients",
        "Alpha Orbital Energies", "Alpha MO coefficients",
        "Beta Orbital Energies", "Beta MO coefficients",
        "Total Energy", "Nuclear charges",
        'Total SCF Density', 'Spin SCF Density',
        'Total MP2 Density', 'Spin MP2 Density',
        'Total MP3 Density', 'Spin MP3 Density',
        'Total CC Density', 'Spin CC Density',
        'Total CI Density', 'Spin CI Density',
        'Mulliken Charges', 'ESP Charges', 'NPA Charges',
        'Polarizability', 'Dipole Moment', 'Quadrupole Moment',
    ])

    # A) Load the geometry
    numbers = fchk["Atomic numbers"]
    coordinates = fchk["Current cartesian coordinates"].reshape(-1, 3)
    pseudo_numbers = fchk["Nuclear charges"]
    # Mask out ghost atoms
    mask = pseudo_numbers != 0.0
    numbers = numbers[mask]
    # Do not overwrite coordinates array, because it is needed to specify basis
    system_coordinates = coordinates[mask]
    pseudo_numbers = pseudo_numbers[mask]

    # B) Load the orbital basis set
    shell_types = fchk["Shell types"]
    shell_map = fchk["Shell to atom map"] - 1
    nprims = fchk["Number of primitives per shell"]
    alphas = fchk["Primitive exponents"]
    ccoeffs_level1 = fchk["Contraction coefficients"]
    ccoeffs_level2 = fchk.get("P(S=P) Contraction coefficients")

    my_shell_types = []
    my_shell_map = []
    my_nprims = []
    my_alphas = []
    con_coeffs = []
    counter = 0
    for i, n in enumerate(nprims):
        if shell_types[i] == -1:
            # Special treatment for SP shell type
            my_shell_types.append(0)
            my_shell_types.append(1)
            my_shell_map.append(shell_map[i])
            my_shell_map.append(shell_map[i])
            my_nprims.append(nprims[i])
            my_nprims.append(nprims[i])
            my_alphas.append(alphas[counter:counter + n])
            my_alphas.append(alphas[counter:counter + n])
            con_coeffs.append(ccoeffs_level1[counter:counter + n])
            con_coeffs.append(ccoeffs_level2[counter:counter + n])
        else:
            my_shell_types.append(shell_types[i])
            my_shell_map.append(shell_map[i])
            my_nprims.append(nprims[i])
            my_alphas.append(alphas[counter:counter + n])
            con_coeffs.append(ccoeffs_level1[counter:counter + n])
        counter += n
    my_shell_types = np.array(my_shell_types)
    my_shell_map = np.array(my_shell_map)
    my_nprims = np.array(my_nprims)
    my_alphas = np.concatenate(my_alphas)
    con_coeffs = np.concatenate(con_coeffs)
    del shell_map
    del shell_types
    del nprims
    del alphas

    obasis = {"centers": coordinates, "shell_map": my_shell_map, "nprims": my_nprims,
              "shell_types": my_shell_types, "alphas": my_alphas, "con_coeffs": con_coeffs}

    # permutation of the orbital basis functions
    permutation_rules = {
        -9: np.arange(19),
        -8: np.arange(17),
        -7: np.arange(15),
        -6: np.arange(13),
        -5: np.arange(11),
        -4: np.arange(9),
        -3: np.arange(7),
        -2: np.arange(5),
        0: np.array([0]),
        1: np.arange(3),
        2: np.array([0, 3, 4, 1, 5, 2]),
        3: np.array([0, 4, 5, 3, 9, 6, 1, 8, 7, 2]),
        4: np.arange(15)[::-1],
        5: np.arange(21)[::-1],
        6: np.arange(28)[::-1],
        7: np.arange(36)[::-1],
        8: np.arange(45)[::-1],
        9: np.arange(55)[::-1],
    }
    permutation = []
    for shell_type in my_shell_types:
        permutation.extend(permutation_rules[shell_type] + len(permutation))
    permutation = np.array(permutation, dtype=int)

    result = {
        'title': fchk['title'],
        'coordinates': system_coordinates,
        'numbers': numbers,
        'obasis': obasis,
        'permutation': permutation,
        'pseudo_numbers': pseudo_numbers,
    }

    nbasis = fchk["Number of basis functions"]

    # C) Load density matrices
    def load_dm(label):
        if label in fchk:
            dm = np.zeros((nbasis, nbasis))
            start = 0
            for i in range(nbasis):
                stop = start + i + 1
                dm[i, :i + 1] = fchk[label][start:stop]
                dm[:i + 1, i] = fchk[label][start:stop]
                start = stop
            return dm

    # First try to load the post-hf density matrices.
    for key in 'MP2', 'MP3', 'CC', 'CI', 'SCF':
        dm_full = load_dm('Total %s Density' % key)
        if dm_full is not None:
            result['dm_full_%s' % key.lower()] = dm_full
        dm_spin = load_dm('Spin %s Density' % key)
        if dm_spin is not None:
            result['dm_spin_%s' % key.lower()] = dm_spin

    # D) Load the wavefunction
    # Handle small difference in fchk files from g03 and g09
    nbasis_indep = fchk.get("Number of independant functions", nbasis)

    # Load orbitals
    nalpha = fchk['Number of alpha electrons']
    nbeta = fchk['Number of beta electrons']
    if nalpha < 0 or nbeta < 0 or nalpha + nbeta <= 0:
        raise ValueError('The file %s does not contain a positive number of electrons.' % filename)
    result['orb_alpha'] = (nbasis, nbasis_indep)
    result['orb_alpha_coeffs'] = np.copy(
        fchk['Alpha MO coefficients'].reshape(nbasis_indep, nbasis).T)
    result['orb_alpha_energies'] = np.copy(fchk['Alpha Orbital Energies'])
    aoccs = np.zeros(nbasis_indep)
    aoccs[:nalpha] = 1.0
    result['orb_alpha_occs'] = aoccs
    if 'Beta Orbital Energies' in fchk:
        # UHF case
        result['orb_beta'] = (nbasis, nbasis_indep)
        result['orb_beta_coeffs'] = np.copy(
            fchk['Beta MO coefficients'].reshape(nbasis_indep, nbasis).T)
        result['orb_beta_energies'] = np.copy(fchk['Beta Orbital Energies'])
        boccs = np.zeros(nbasis_indep)
        boccs[:nbeta] = 1.0
        result['orb_beta_occs'] = boccs

    elif fchk['Number of beta electrons'] != fchk['Number of alpha electrons']:
        # ROHF case
        result['orb_beta'] = (nbasis, nbasis_indep)
        result['orb_beta_coeffs'] = fchk['Alpha MO coefficients'].reshape(nbasis_indep, nbasis).T
        result['orb_beta_energies'] = fchk['Alpha Orbital Energies']
        boccs = np.zeros(nbasis_indep)
        boccs[:nbeta] = 1.0
        result['orb_beta_occs'] = boccs

        # Delete dm_full_scf because it is known to be buggy
        result.pop('dm_full_scf')

    # E) Load properties
    result['energy'] = fchk['Total Energy']
    if 'Polarizability' in fchk:
        result['polar'] = _triangle_to_dense(fchk['Polarizability'])
    if 'Dipole Moment' in fchk:
        result['dipole_moment'] = fchk['Dipole Moment']
    if 'Quadrupole Moment' in fchk:
        # Convert to HORTON ordering: xx, xy, xz, yy, yz, zz
        result['quadrupole_moment'] = fchk['Quadrupole Moment'][[0, 3, 4, 1, 5, 2]]

    # F) Load optional properties
    # Mask out ghost atoms from charges
    if 'Mulliken Charges' in fchk:
        result['mulliken_charges'] = fchk['Mulliken Charges'][mask]
    if 'ESP Charges' in fchk:
        result['esp_charges'] = fchk['ESP Charges'][mask]
    if 'NPA Charges' in fchk:
        result['npa_charges'] = fchk['NPA Charges'][mask]

    return result


def _load_fchk_low(lit: LineIterator, labels: List[str] = None) -> Dict:
    """Read selected fields from a formatted checkpoint file."""
    result = {}
    # Read the two-line header
    result['title'] = next(lit).strip()
    words = next(lit).split()
    if len(words) == 3:
        result['command'], result['lot'], result['obasis'] = words
    elif len(words) == 2:
        result['command'], result['lot'] = words
    else:
        lit.error('The second line of the FCHK file should contain two or three words.')

    # Read all fields, go all they way until the until unless all requested
    # labels are used.
    if labels is not None:
        labels = set(labels)
    while labels is None or len(labels) > 0:
        try:
            label, value = _load_fchk_field(lit, labels)
        except StopIteration:
            # We read the whole file, this happens when more labels are given
            # than those present in the file, which should be allowed.
            break
        result[label] = value
    return result


def _load_fchk_field(lit: LineIterator, labels: List[str]) -> Tuple[str, object]:
    """Read a single field with one of the given labels."""
    while True:
        # find a sane header line
        line = next(lit)
        label = line[:43].strip()
        words = line[43:].split()
        if len(words) == 0:
            continue
        if words[0] == 'I':
            datatype = int
        elif words[0] == 'R':
            datatype = float
        else:
            continue
        if labels is not None and label not in labels:
            continue
        if len(words) == 2:
            try:
                return label, datatype(words[1])
            except ValueError:
                lit.error("Could not interpret: {}".format(words[1]))
        elif len(words) == 3:
            if words[1] != "N=":
                lit.error("Expected N= not found.")
            length = int(words[2])
            value = np.zeros(length, datatype)
            counter = 0
            words = []
            while counter < length:
                if len(words) == 0:
                    words = next(lit).split()
                word = words.pop(0)
                try:
                    value[counter] = datatype(word)
                except (ValueError, OverflowError):
                    lit.error('Could not interpret: {}'.format(word))
                counter += 1
            return label, value


def _triangle_to_dense(triangle: np.ndarray) -> np.ndarray:
    """Convert a symmetric matrix in triangular storage to a dense square matrix.

    Parameters
    ----------
    triangle
        A row vector containing all the unique matrix elements of symmetric
        matrix. (Either the lower-triangular part in row major-order or the
        upper-triangular part in column-major order.)

    Returns
    -------
    ndarray
        a square symmetric matrix.

    """
    nrow = int(np.round((np.sqrt(1 + 8 * len(triangle)) - 1) / 2))
    result = np.zeros((nrow, nrow))
    begin = 0
    for irow in range(nrow):
        end = begin + irow + 1
        result[irow, :irow + 1] = triangle[begin:end]
        result[:irow + 1, irow] = triangle[begin:end]
        begin = end
    return result
