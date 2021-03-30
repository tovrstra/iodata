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
"""CAR File format.

This is a file format used by Materials Studio.
"""

import datetime
from typing import TextIO

import numpy as np

from ..docstrings import document_load_one, document_dump_one
from ..iodata import IOData
from ..periodic import sym2num, num2sym
from ..utils import angstrom, LineIterator


__all__ = []


PATTERNS = ['*.car']


@document_load_one("CAR", ['atcoords', 'atnums', 'atffparams', 'extra'], ['title'])
def load_one(lit: LineIterator) -> dict:
    """Do not edit this docstring. It will be overwritten."""
    title = None
    atnames = []
    atcoords = []
    restypes = []
    resnums = []
    attypes = []
    atnums = []
    charges = []
    while True:
        try:
            line = next(lit)
        except StopIteration:
            break
        # Parse non-atom lines.
        if line.startswith("!") or line.startswith("PBC"):
            # Parsing periodic boundary conditions is not supported yet.
            continue
        if line == "end\n":
            break
        if title is None:
            title = line.strip()
            continue
        # All non-atom lines are handled by the previous conditionals.
        atnames.append(line[:5].strip())
        atcoords.append([float(line[6:20]), float(line[21:35]), float(line[36:50])])
        restypes.append(line[51:55])
        resnums.append(int(line[56:62]))
        attypes.append(line[63:69].strip())
        atnums.append(sym2num.get(line[70:72].strip().title()))
        charges.append(float(line[73:]))
    atffparams = {
        "restypes": np.array(restypes),
        "resnums": np.array(resnums),
        "attypes": np.array(attypes),
        "charges": np.array(charges),
    }
    return {
        "atnums": np.array(atnums),
        "atcoords": np.array(atcoords) * angstrom,
        "atffparams": atffparams,
        "title": title,
        "extra": {"atnames": np.array(atnames)}
    }


@document_dump_one("CAR", ['atcoords', 'atnums'], ['atffparams', 'title', 'extra'])
def dump_one(f: TextIO, data: IOData):
    """Do not edit this docstring. It will be overwritten."""
    # Write header.
    print("!BIOSYM archive 3", file=f)
    print("PBC=OFF", file=f)
    print(data.title or "Created with IOData", file=f)
    print("!DATE {:%c}".format(datetime.datetime.now()), file=f)
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
    # Write atom lines.
    for iatom in range(data.natom):
        print("{:<5s} {:14.9f} {:14.9f} {:14.9f} {:<4s} {:<6d} {:<6s} {:>2s} {:7.3f}".format(
            atnames[iatom],
            data.atcoords[iatom, 0] / angstrom,
            data.atcoords[iatom, 1] / angstrom,
            data.atcoords[iatom, 2] / angstrom,
            restypes[iatom],
            resnums[iatom],
            attypes[iatom],
            num2sym[data.atnums[iatom]],
            atcharges[iatom],
        ), file=f)
    # Write footer.
    print("end", file=f)
    print("end", file=f)
