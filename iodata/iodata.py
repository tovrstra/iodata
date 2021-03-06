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
# pragma pylint: disable=wrong-import-order,invalid-name,redefined-outer-name,too-many-branches
"""Module for handling input/output from different file formats."""


import os
import numpy as np

from typing import List, Tuple, Type
from fnmatch import fnmatch
from pkgutil import iter_modules
from importlib import import_module

from .utils import LineIterator


__all__ = ['IOData']


format_modules = []
for module_info in iter_modules(import_module('iodata.formats').__path__):
    if not module_info.ispkg:
        format_module = import_module('iodata.formats.' + module_info.name)
        if hasattr(format_module, 'patterns'):
            format_modules.append(format_module)


class ArrayTypeCheckDescriptor:
    def __init__(self, name: str, ndim: int = None, shape: Tuple = None, dtype: Type = None,
                 matching: List[str] = None, default: str = None, doc=None):
        """Initialize decorator to perform type and shape checking of np.ndarray attributes.

        Parameters
        ----------
        name
            Name of the attribute (without leading underscores).
        ndim
            The number of dimensions of the array.
        shape
            The shape of the array. Use -1 for dimensions where the shape is
            not fixed a priori.
        dtype
            The datatype of the array.
        matching
            A list of names of other attributes that must have consistent
            shapes. This argument requires that the shape is specified.
            All dimensions for which the shape tuple equals -1 are must be
            the same in this attribute and the matching attributes.
        default
            The name of another (type-checked) attribute to return as default
            when this attribute is not set

        """
        if matching is not None and shape is None:
            raise TypeError('The matching argument requires the shape to be specified.')

        self._name = name
        self._ndim = ndim
        self._shape = shape
        if dtype is None:
            self._dtype = None
        else:
            self._dtype = np.dtype(dtype)
        self._matching = matching
        self._default = default
        self.__doc__ = doc or 'A type-checked attribute'

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        if self._default is not None and not hasattr(obj, '_' + self._name):
            setattr(obj, '_' + self._name, (getattr(obj, '_' + self._default).astype(self._dtype)))
        return getattr(obj, '_' + self._name)

    def __set__(self, obj, value):
        # try casting to proper dtype:
        value = np.array(value, dtype=self._dtype, copy=False)
        # if not isinstance(value, np.ndarray):
        #    raise TypeError('Attribute \'%s\' of \'%s\' must be a numpy '
        #                    'array.' % (self._name, type(obj)))
        if self._ndim is not None and value.ndim != self._ndim:
            raise TypeError(f"Attribute '{self._name}' of '{type(obj)}' must be a numpy array "
                            f"with {self._ndim} dimension(s).")
        if self._shape is not None:
            for i in range(len(self._shape)):
                if self._shape[i] >= 0 and self._shape[i] != value.shape[i]:
                    raise TypeError(f"Attribute '{self._name}' of '{type(obj)}' must be a numpy"
                                    f" array {self._shape[i]} elements in dimension {i}.")
        if self._dtype is not None:
            if not issubclass(value.dtype.type, self._dtype.type):
                raise TypeError(f"Attribute '{self._name}' of '{type(obj)}' must be a numpy "
                                f"array with dtype '{self.dtype.type}'.")
        if self._matching is not None:
            for othername in self._matching:
                other = getattr(obj, '_' + othername, None)
                if other is not None:
                    for i in range(len(self._shape)):
                        if self._shape[i] == -1 and \
                                other.shape[i] != value.shape[i]:
                            raise TypeError(f"shape[{i}] of attribute '{self._name}' of "
                                            f"'{type(obj)}' in is incompatible with "
                                            f"that of '{othername}'.")
        setattr(obj, '_' + self._name, value)

    def __delete__(self, obj):
        delattr(obj, '_' + self._name)


class IOData:
    """A container class for data loaded from (or to be written to) a file.

       In principle, the constructor accepts any keyword argument, which is
       stored as an attribute. All attributes are optional. Attributes can be
       set are removed after the IOData instance is constructed. The following
       attributes are supported by at least one of the io formats:

       **Type checked array attributes (if present):**

       cube_data
            A (L, M, N) array of data on a uniform grid (defined by ugrid).

       coordinates
            A (N, 3) float array with Cartesian coordinates of the atoms.

       numbers
            A (N,) int vector with the atomic numbers.

       polar
            A (3, 3) matrix containing the dipole polarizability tensor.

       pseudo_numbers
            A (N,) float array with pseudo-potential core charges.

       **Unspecified type (duck typing):**

       cell
            A Cell object that describes the (generally triclinic) periodic
            boundary conditions.

       core_energy
            The Hartree-Fock energy due to the core orbitals

       energy
            The total energy (electronic+nn)

       er
            The electron repulsion four-index operator

       orb_alpha
            The alpha orbitals (coefficients, occupations and energies).

       orb_beta
            The beta orbitals (coefficients, occupations and energies).

       esp_charges
            Charges fitted to the electrostatic potential

       dm_full (optionally with a suffix like _mp2, _mp3, _cc, _ci, _scf).
            The spin-summed first-order density matrix.

       dm_spin (optionally with a suffix like _mp2, _mp3, _cc, _ci, _scf).
            The spin-difference first-order density matrix.

       grid
            An integration grid (usually a UniformGrid instance).

       kin
            The kinetic energy operator.

       links
            A mapping between the atoms in the primitive unit and the
            crystallographic unit.

       ms2
            The spin multiplicity.

       mulliken_charges
            Mulliken AIM charges.

       na
            The nuclear attraction operator.

       nelec
            The number of electrons.

       npa_charges
            Natural charges.

       obasis
            An OrderedDict containing parameters to instantiate a GOBasis class.

       olp
            The overlap operator.

       one_mo
            One-electron integrals in the (Hartree-Fock) molecular-orbital basis

       permutation
            The permutation applied to the basis functions.

       signs
            The sign changes applied to the basis functions.

       title
            A suitable name for the data.

       two_mo
            Two-electron integrals in the (Hartree-Fock) molecular-orbital basis

    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    # Numpy array attributes that may require orbital basis reordering or sign correction.
    # Note: this list is a very fragile thing and should be implemented differently in
    # future. Ideally, each IO format should be implemented as a class, with a load and
    # a dump method. Besides these two basic methods, it should also provide additional
    # information on the fields it can read/write and which should be considered for
    # reordering basis functions or changing their signs. The current approach to maintain
    # this list at the iodata level requires us to keep it up-to-date whenever we change
    # something in the file formats. (The same can be said of the class doc string and the
    # documentation of the file formats.)
    two_index_names = ['dm_full', 'dm_full_mp2', 'dm_full_mp3', 'dm_full_cc',
                       'dm_full_ci', 'dm_full_scf', 'dm_spin', 'dm_spin_mp2',
                       'dm_spin_mp3', 'dm_spin_cc', 'dm_spin_ci', 'dm_spin_scf', 'kin',
                       'na', 'olp']

    # only perform type checking on minimal attributes
    numbers = ArrayTypeCheckDescriptor('numbers', 1, (-1,), int, ['coordinates', 'pseudo_numbers'],
                                       doc="A (N,) int vector with the atomic numbers.")
    coordinates = ArrayTypeCheckDescriptor('coordinates', 2, (-1, 3), float,
                                           ['numbers', 'pseudo_numbers'],
                                           doc="A (N, 3) float array with Cartesian coordinates "
                                               "of the atoms.")
    cube_data = ArrayTypeCheckDescriptor('cube_data', 3,
                                         doc="A (L, M, N) array of data on a uniform grid "
                                             "(defined by ugrid).")
    polar = ArrayTypeCheckDescriptor('polar', 2, (3, 3), float,
                                     doc="A (3, 3) matrix containing the dipole polarizability "
                                         "tensor.")
    pseudo_numbers = ArrayTypeCheckDescriptor('pseudo_numbers', 1, (-1,), float,
                                              ['coordinates', 'numbers'], 'numbers',
                                              doc="A (N,) float array with pseudo-potential core "
                                                  "charges.")

    @property
    def natom(self) -> int:
        """The number of atoms"""
        if hasattr(self, 'numbers'):
            return len(self.numbers)
        elif hasattr(self, 'coordinates'):
            return len(self.coordinates)
        elif hasattr(self, 'pseudo_numbers'):
            return len(self.pseudo_numbers)


def load_one(*filenames: str) -> IOData:
    """Load data from a file.

    This routine uses the extension or prefix of the filename to
    determine the file format. It returns a dictionary with data loaded
    from the file.

    For each file format, a specialized function is called that returns a
    dictionary with data from the file.

    Parameters
    ----------
    filenames : str or sequence of str
        The files to load data from. When multiple files are given, data
        from the first file is overwritten by data from the second, etc.
        When one file contains sign and permutation changes for the
        orbital basis, these changes will be applied to data from all
        other files.

    Returns
    -------
    out : IOData
        The instance of IOData with data loaded from the input files.

    """
    result = {}
    for filename in filenames:
        basename = os.path.basename(filename)
        for format_module in format_modules:
            if any(fnmatch(basename, pattern) for pattern in format_module.patterns):
                lit = LineIterator(filename)
                try:
                    result.update(format_module.load(lit))
                except StopIteration:
                    raise lit.error("File ended before all data was read.")
                break
        else:
            raise ValueError('Unknown file format for reading: %s' % filename)

    # Apply changes in atomic orbital basis order
    permutation = result.get('permutation')
    if permutation is not None:
        for name in IOData.two_index_names:
            value = result.get(name)
            if value is not None:
                value[:] = value[permutation][:, permutation]
        er = result.get('er')
        if er is not None:
            er[:] = er[permutation][:, permutation][:, :, permutation][:, :, :, permutation]
        orb_alpha_coeffs = result.get('orb_alpha_coeffs')
        if orb_alpha_coeffs is not None:
            orb_alpha_coeffs[:] = orb_alpha_coeffs[permutation]
        orb_beta_coeffs = result.get('orb_beta_coeffs')
        if orb_beta_coeffs is not None:
            orb_beta_coeffs[:] = orb_beta_coeffs[permutation]
        del result['permutation']

    # Apply changes in atomic orbital basis sign conventions
    signs = result.get('signs')
    if signs is not None:
        for name in IOData.two_index_names:
            value = result.get(name)
            if value is not None:
                value *= signs
                value *= signs.reshape(-1, 1)
        er = result.get('er')
        if er is not None:
            er *= signs
            er *= signs.reshape(-1, 1)
            er *= signs.reshape(-1, 1, 1)
            er *= signs.reshape(-1, 1, 1, 1)
        orb_alpha_coeffs = result.get('orb_alpha_coeffs')
        if orb_alpha_coeffs is not None:
            orb_alpha_coeffs *= signs.reshape(-1, 1)
        orb_beta_coeffs = result.get('orb_beta_coeffs')
        if orb_beta_coeffs is not None:
            orb_beta_coeffs *= signs.reshape(-1, 1)
        del result['signs']

    return IOData(**result)


def dump_one(iodata: IOData, filename: str):
    """Write data to a file.

    This routine uses the extension or prefix of the filename to determine
    the file format. For each file format, a specialized function is
    called that does the real work.

    Parameters
    ----------
    iodata
        The object containing the data to be written.
    filename : str
        The file to write the data to.

    """
    basename = os.path.basename(filename)
    for format_module in format_modules:
        if any(fnmatch(basename, pattern)for pattern in format_module.patterns):
            format_module.dump(filename, iodata)
            break
    else:
        raise ValueError('Unknown file format for writing: %s' % filename)
