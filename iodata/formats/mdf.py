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
"""MDF File format.

This is a file format used by Materials Studio.

This implementation may not support all features of the MDF format. It is
primarily written to work well with msi2lammps. It is assumed that the atom
lines in this file format adhere to the column in the MDF_HEADER variable below.

This implementation is derived from a few example files. It may therefore not
yet work on files who deviate significantly from these examples.

"""

import datetime
from typing import TextIO

import numpy as np

from ..docstrings import document_load_one, document_dump_one
from ..iodata import IOData
from ..periodic import sym2num, num2sym
from ..utils import LineIterator


__all__ = []


PATTERNS = ['*.mdf']


def _load_mdf_base(lit):
    """Load the mdf data into a Python representation."""
    sections = {}
    my_section = None
    while True:
        try:
            line = next(lit).strip()
        except StopIteration:
            break
        if line.startswith("!") or line == "":
            continue
        if line == "#end":
            break
        if line.startswith("#"):
            my_section = []
            sections[line[1:]] = my_section
        elif line.startswith("@"):
            my_section.append([line[1:], []])
        else:
            my_section[-1][1].append(line)
    return sections


EXPECTED_COLUMNS = [
    "element", "atom_type", "charge_group", "isotope", "formal_charge",
    "charge", "switching_atom", "oop_flag", "chirality_flag", "occupancy",
    "xray_temp_factor", "connections"
]


@document_load_one("MDF", ['atnums', 'atffparams', 'extra'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    sections = _load_mdf_base(lit)
    # Verify the column definitions and extract the molecule
    icol = 0
    for field, lines in sections["topology"]:
        if field.startswith("column"):
            if icol >= len(EXPECTED_COLUMNS):
                lit.error("Too many columns defined in MDF file.")
            column_name = EXPECTED_COLUMNS[icol]
            if field != f"column {icol + 1} {column_name}":
                lit.error("Columns in MDF file deviate from the expected format.")
            icol += 1
        elif field.startswith("molecule"):
            title = field[8:].strip()
            mol_lines = lines
    # Parse the molecule lines
    keys = []  # only used for interpreting connections
    restypes = []
    resnums = []
    atnames = []
    atnums = []
    attypes = []
    atcharges = []
    occupancies = []
    bfactors = []
    neighbors = []
    for line in mol_lines:
        key = line[:18].strip()
        keys.append(key)
        res, atname = key.split(":")
        restype, resnum = res.split("_")
        restypes.append(restype)
        resnums.append(int(resnum))
        atnames.append(atname)
        atnums.append(sym2num[line[19:21].strip().title()])
        attypes.append(line[22:25].strip())
        atcharges.append(float(line[42:52]))
        occupancies.append(float(line[59:65]))
        bfactors.append(float(line[66:73]))
        other_keys = []
        for other_atname in line[74:].split():
            other_keys.append(f"{res}:{other_atname}")
        neighbors.append(other_keys)
    # Parse connectivity into a bond matrix
    keys2iatom = dict((key, iatom) for iatom, key in enumerate(keys))
    bonds = []
    for iatom0, other_keys in enumerate(neighbors):
        for other_key in other_keys:
            iatom1 = keys2iatom[other_key]
            if iatom1 > iatom0:
                bonds.append([iatom0, iatom1, 1])
    bonds = np.array(bonds) if len(bonds) > 0 else None
    # Wrap up
    return {
        "atnums": np.array(atnums),
        "bonds": bonds,
        "atffparams": {
            "attypes": np.array(attypes),
            "charges": np.array(atcharges),
            "resnums": np.array(resnums),
            "restypes": np.array(restypes),
        },
        "title": title,
        "extra": {
            "occupancies": np.array(occupancies),
            "bfactors": np.array(bfactors),
            "atnames": np.array(atnames),
        }
    }


MDF_HEADER = """\
!BIOSYM molecular_data 4

!Date: {:%c}   IOData Generated MDF file

#topology

@column 1 element
@column 2 atom_type
@column 3 charge_group
@column 4 isotope
@column 5 formal_charge
@column 6 charge
@column 7 switching_atom
@column 8 oop_flag
@column 9 chirality_flag
@column 10 occupancy
@column 11 xray_temp_factor
@column 12 connections

"""


@document_dump_one("MDF", ['atnums'], ['atffparams', 'title', 'extra', 'bonds'])
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    # Write header.
    f.write(MDF_HEADER.format(datetime.datetime.now()))
    print("@molecule", data.title, file=f)
    print(file=f)
    # Generate default values if some fields are not present
    atnames = data.extra.get("atnames")
    if atnames is None:
        atnames = [num2sym[atnum] + str(idx + 1) for idx, atnum in enumerate(data.atnums)]
    restypes = data.atffparams.get("restypes")
    if restypes is None:
        restypes = ["XXXX"] * data.natom
    resnums = data.atffparams.get("resnums")
    if resnums is None:
        resnums = np.ones(data.natom, dtype=int)
    atcharges = data.atffparams.get("charges")
    if atcharges is None:
        atcharges = np.zeros(data.natom)
    attypes = data.atffparams.get("attypes")
    if attypes is None:
        attypes = [num2sym[atnum] for atnum in data.atnums]
    occupancies = data.atffparams.get("occupancies")
    if occupancies is None:
        occupancies = np.ones(data.natom)
    bfactors = data.extra.get("bfactors")
    if bfactors is None:
        bfactors = np.zeros(data.natom)
    # Convert bonds to connections. This will ignore issues with duplicated
    # atom names, residue mismatches, etc. Detecting all of these issues goes
    # beyond the scope of this implementation.
    connections = [[] for iatom in range(data.natom)]
    if data.bonds is not None:
        for iatom0, iatom1 in data.bonds[:, :2]:
            connections[iatom0].append(atnames[iatom1])
            connections[iatom1].append(atnames[iatom0])
    # Write atom lines.
    for iatom in range(data.natom):
        key = "{}_{}:{}".format(restypes[iatom], resnums[iatom], atnames[iatom])
        print("{:<18s} {:>2s} {:>3s}      1     0  0 {:10.4f} 0 0 8 {:6.4f} {:7.4f} {}".format(
            key,
            num2sym[data.atnums[iatom]],
            attypes[iatom],
            atcharges[iatom],
            occupancies[iatom],
            bfactors[iatom],
            " ".join(connections[iatom]),
        ), file=f)
    # Write footer.
    print(file=f)
    print("#end", file=f)
