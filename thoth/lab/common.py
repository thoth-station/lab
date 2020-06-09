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
import json
import os
from thoth.storages.result_base import ResultStorageBase
from typing import Optional, Union
from pathlib import Path
from zipfile import ZipFile

_LOGGER = logging.getLogger("thoth.lab.common")


def extract_zip_file(file_path: Path):
    """Extract files from zip files."""
    with ZipFile(file_path, "r") as zip_file:
        zip_file.printdir()

        _LOGGER.debug("Extracting all the files now...")
        zip_file.extractall()


def _aggregate_thoth_results(
    limit_results: bool = False,
    max_ids: int = 5,
    is_local: bool = True,
    repo_path: Optional[Path] = None,
    store: Optional[ResultStorageBase] = None,
    is_inspection: Optional[str] = None,
) -> Union[list, dict]:
    """Aggregate results from jsons stored in Ceph for Thoth or locally from repo.

    :param limit_results: reduce the number of reports ids considered to `max_ids` to test analysis
    :param max_ids: maximum number of reports ids considered
    :param is_local: flag to retreive the dataset locally or from S3 (credentials are required)
    :param repo_path: required if you want to retrieve the dataset locally and `is_local` is set to True
    :param store: ResultStorageBase type depending on Thoth data (e.g solver, performance, adviser, etc.)
    :param is_inspection: flag used only for InspectionResultStore as we store results in batches
    """
    if limit_results:
        _LOGGER.debug(f"Limiting results to {max_ids} to test functions!!")

    if is_inspection:
        files = {}
    else:
        files = []

    counter = 1

    if not is_local:
        store = store()
        store.connect()

        for document_id in store.get_document_listing():
            _LOGGER.debug("Document n. %r", counter)
            _LOGGER.debug(document_id)

            report = store.retrieve_document(document_id=document_id)

            files.append(report)

            counter += 1

    elif is_local:
        _LOGGER.debug(f"Retrieving dataset at path... {repo_path}")
        if not repo_path.exists():
            raise Exception("There is no dataset at this path")

        for file_path in repo_path.iterdir():
            _LOGGER.debug(file_path)

            if os.path.isdir(file_path) and is_inspection:
                main_repo = file_path
                files[str(main_repo)] = []

                for file_path in main_repo.iterdir():
                    if "specification" in str(file_path):
                        with open(file_path, "r") as json_file_type:
                            specification = json.load(json_file_type)
                        break

                if specification:
                    for file_path in main_repo.iterdir():
                        if "specification" not in str(file_path):
                            with open(file_path, "r") as json_file_type:
                                json_file = json.load(json_file_type)
                                json_file["requirements"] = specification["python"]["requirements"]
                                json_file["requirements_locked"] = specification["python"]["requirements_locked"]
                                # json_file["runtime_environment"] = specification["python"]["runtime_environment"]
                                # pop build logs to save some memory (not necessary for now)
                                json_file["build_log"] = None

                            json_file["identifier"] = main_repo.stem
                            files[str(main_repo)].append(json_file)
                            counter += 1

            else:

                with open(file_path, "r") as json_file_type:
                    json_file = json.load(json_file_type)

                files.append(json_file)

                counter += 1

    _LOGGER.debug("Number of file retrieved is: %r" % counter)

    return files
