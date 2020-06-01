#!/usr/bin/env python3
# thoth-lab
# Copyright(C) 2020 Francesco Murdaca
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Common methods for thoth-lab."""

import logging
from pathlib import Path
from zipfile import ZipFile

_LOGGER = logging.getLogger("thoth.lab.common")


def extract_zip_file(file_path: Path):
    """Extract files from zip files."""
    with ZipFile(file_path, "r") as zip_file:
        zip_file.printdir()

        _LOGGER.debug("Extracting all the files now...")
        zip_file.extractall()
