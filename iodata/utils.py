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
"""Utility functions module."""


import numpy as np
import scipy.constants as spc

from scipy.linalg import eigh
from typing import List, Dict, Tuple, NamedTuple

from .overlap import get_shell_nbasis


__all__ = ['LineIterator', 'set_four_index_element', 'MolecularOrbitals']


angstrom = spc.angstrom / spc.value(u'atomic unit of length')
electronvolt = 1. / spc.value(u'hartree-electron volt relationship')


class LineIterator:
    """Iterator class for looping over lines and keeping track of the line number."""

    def __init__(self, filename: str):
        """Initialize a LineIterator.

        Parameters
        ----------
        filename
            The file that will be read.

        """
        self.filename = filename
        self._f = open(filename)
        self.lineno = 0
        self.stack = []

    def __del__(self):
        self._f.close()

    def __iter__(self):
        return self

    def __next__(self):
        """Get the next line, will also increase the lineno attribute."""
        if len(self.stack) > 0:
            line = self.stack.pop()
        else:
            line = next(self._f)
        self.lineno += 1
        return line

    def error(self, msg):
        """Raise an error while reading a file.

        Filename and line number are added to the message.
        """
        raise IOError("{}:{} {}".format(self.filename, self.lineno, msg))

    def back(self, line):
        """Push one line back, which will be read again when next(lit) is called."""
        self.stack.append(line)
        self.lineno -= 1


class MolecularOrbitals(NamedTuple):
    """Molecular Orbitals Class.

    Attributes
    ----------
    type : str
        Molecular orbital type; choose from 'restricted', 'unrestricted', or 'generalized'.
    norb_a : int
        Number of alpha molecular orbitals.
    norb_b : int
        Number of beta molecular orbitals.
    occs : np.ndarray
        Molecular orbital occupation numbers.
    coeffs : np.ndarray
        Molecular orbital basis coefficients.
    irreps : np.ndarray
        Irreducible representation.
    energies : np.ndarray
        Molecular orbital energies.

    """

    type: str
    norb_a: int
    norb_b: int
    occs: np.ndarray
    coeffs: np.ndarray
    irreps: np.ndarray
    energies: np.ndarray


def str_to_shell_types(s: str, pure: bool = False) -> List[int]:
    """Convert a string into a list of contraction types"""
    if pure:
        d = {'s': 0, 'p': 1, 'd': -2, 'f': -3, 'g': -4, 'h': -5, 'i': -6}
    else:
        d = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}
    return [d[c] for c in s.lower()]


def shell_type_to_str(shell_type: np.ndarray) -> Dict:
    """Convert a shell type into a character"""
    return {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h', 6: 'i'}[abs(shell_type)]


def set_four_index_element(four_index_object: np.ndarray, i: int, j: int, k: int, l: int,
                           value: float):
    """Assign values to a four index object, account for 8-fold index symmetry.

    This function assumes physicists' notation

    Parameters
    ----------
    four_index_object
        The four-index object. It will be written to.
        shape=(nbasis, nbasis, nbasis, nbasis), dtype=float
    i, j, k, l
        The indices to assign to.
    value
        The value of the matrix element to store.
    """
    four_index_object[i, j, k, l] = value
    four_index_object[j, i, l, k] = value
    four_index_object[k, j, i, l] = value
    four_index_object[i, l, k, j] = value
    four_index_object[k, l, i, j] = value
    four_index_object[l, k, j, i] = value
    four_index_object[j, k, l, i] = value
    four_index_object[l, i, j, k] = value


def shells_to_nbasis(shell_types: np.ndarray) -> int:
    nbasis_shell = [get_shell_nbasis(i) for i in shell_types]
    return sum(nbasis_shell)


def volume(rvecs: np.ndarray) -> float:
    """Calculates cell volume

    Parameters
    ----------
    rvecs
        a numpy matrix of shape (x,3) where x is in {1,2,3}
    """
    nvecs = rvecs.shape[0]
    if len(rvecs.shape) == 1 or nvecs == 1:
        return np.linalg.norm(rvecs)
    elif nvecs == 2:
        return np.linalg.norm(np.cross(rvecs[0], rvecs[1]))
    elif nvecs == 3:
        return np.linalg.det(rvecs)
    else:
        raise ValueError("Argument rvecs should be of shape (x, 3), where x is in {1, 2, 3}")


def derive_naturals(dm: np.ndarray, overlap: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """Derive natural orbitals from a given density matrix and assign the result to self.

    Parameters
    ----------
    dm
        The density matrix.
        shape=(nbasis, nbasis)
    overlap
        The overlap matrix
        shape=(nbasis, nbasis)

    Returns
    -------
    coeffs
        Orbital coefficients
        shape=(nbasis, nfn)
    occs
        Orbital occupations
        shape=(nfn, )
    energies
        Orbital energies
        shape=(nfn, )
    """
    # Transform density matrix to Fock-like form
    sds = np.dot(overlap.T, np.dot(dm, overlap))
    # Diagonalize and compute eigenvalues
    evals, evecs = eigh(sds, overlap)
    coeffs = np.zeros_like(overlap)
    coeffs = evecs[:, :coeffs.shape[1]]
    occs = evals
    energies = np.zeros(overlap.shape[0])
    return coeffs, occs, energies


def check_dm(dm: np.ndarray, overlap: np.ndarray, eps: float = 1e-4, occ_max: float = 1.0):
    """Check if the density matrix has eigenvalues in the proper range.

    Parameters
    ----------
    dm
        The density matrix
        shape=(nbasis, nbasis), dtype=float
    overlap
        The overlap matrix
        shape=(nbasis, nbasis), dtype=float
    eps
        The threshold on the eigenvalue inequalities.
    occ_max
        The maximum occupation.

    Raises
    ------
    ValueError
        When the density matrix has wrong eigenvalues.
    """
    # construct natural orbitals
    coeffs, occupations, energies = derive_naturals(dm, overlap)
    if occupations.min() < -eps:
        raise ValueError('The density matrix has eigenvalues considerably smaller than '
                         'zero. error=%e' % (occupations.min()))
    if occupations.max() > occ_max + eps:
        raise ValueError('The density matrix has eigenvalues considerably larger than '
                         'max. error=%e' % (occupations.max() - 1))
