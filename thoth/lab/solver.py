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

"""Solver results processing and analysis."""

import logging
import json

import pandas as pd

from pathlib import Path

from thoth.storages import SolverResultsStore

_LOGGER = logging.getLogger("thoth.lab.solver")


def aggregate_solver_results(
    limit_results: bool = False, max_ids: int = 5, is_local: bool = True, solver_repo_path: Path = Path("solver")
) -> list:
    """Aggregate solver results from jsons stored in Ceph or locally from `solver` repo.

    :param limit_results: reduce the number of solver reports ids considered to `max_ids` to test analysis
    :param max_ids: maximum number of solver reports ids considered
    :param is_local: flag to retreive the dataset locally or from S3 (credentials are required)
    :param solver_repo_path: required if you want to retrieve the solver dataset locally and `is_local` is set to True
    """
    if limit_results:
        _LOGGER.debug(f"Limiting results to {max_ids} to test functions!!")

    solver_reports = []
    c_s = 1

    if not is_local:
        solver_store = SolverResultsStore()
        solver_store.connect()

        for solver_id in solver_store.get_document_listing():
            _LOGGER.debug("Document n. %r", c_s)
            _LOGGER.debug(solver_id)

            solver_report = solver_store.retrieve_document(document_id=solver_id)

            solver_reports.append(solver_report)

            c_s += 1

    elif is_local:
        _LOGGER.debug(f"Retrieving solver dataset at path... {solver_repo_path}")
        if not solver_repo_path.exists():
            raise Exception("There is no dataset at this path")

        for solver_document_path in solver_repo_path.iterdir():
            _LOGGER.debug(solver_document_path)

            with open(solver_document_path, "r") as solver_json:
                solver_report = json.load(solver_json)

            solver_reports.append(solver_report)
        c_s += 1

    _LOGGER.debug("Number of Solver reports retrieved is: %r" % c_s)

    return solver_reports


def construct_solver_from_metadata(solver_report_metadata: dict) -> str:
    """Construct solver from solver report metadata."""
    os_name = solver_report_metadata["os_release"]["name"].lower()
    os_version = "".join([l for l in solver_report_metadata["os_release"]["version"] if l.isdigit()])
    python_interpreter = f'{solver_report_metadata["python"]["major"]}{solver_report_metadata["python"]["minor"]}'
    solver = f"{os_name}-{os_version}-py{python_interpreter}"

    return solver


def extract_data_from_solver_metadata(solver_report_metadata: dict) -> dict:
    """Extract data from solver report metadata."""
    solver = construct_solver_from_metadata(solver_report_metadata)
    solver_parts = solver.split("-")

    requirements = solver_report_metadata["arguments"]["python"]["requirements"]

    extracted_metadata = {
        "document_id": solver_report_metadata["document_id"],
        "datetime": solver_report_metadata["datetime"],
        "requirements": requirements,
        "solver": solver,
        "os_name": solver_parts[0],
        "os_version": solver_parts[1],
        "python_interpreter": ".".join(solver_parts[2][2:]),
        "analyzer_version": solver_report_metadata["analyzer_version"],
    }

    return extracted_metadata


def extract_tree_from_solver_result(solver_report_result: dict) -> list:
    """Extract data from solver report result."""
    packages = []
    for python_package_info in solver_report_result["tree"]:
        package = {
            "package_name": python_package_info["package_name"],
            "package_version": python_package_info["package_version_requested"],
            "index_url": python_package_info["index_url"],
            "importlib_metadata": python_package_info["importlib_metadata"]["metadata"],
            "dependencies": python_package_info["dependencies"],
        }
        packages.append(package)

    return packages


def extract_errors_from_solver_result(solver_report_result_errors: list) -> list:
    """Extract all errors from solver report (if any)."""
    errors = []
    for error in solver_report_result_errors:
        errors.append(
            {
                "package_name": error["package_name"],
                "package_version": error["package_version"],
                "index_url": error["index_url"],
                "type": error["type"],
                "command": error["details"]["command"] if "command" in error["details"] else None,
                "message": error["details"]["message"] if "message" in error["details"] else None,
                "return_code": error["details"]["return_code"] if "return_code" in error["details"] else None,
                "stderr": error["details"]["stderr"] if "stderr" in error["details"] else None,
                "stdout": error["details"]["stdout"] if "stdout" in error["details"] else None,
                "timeout": error["details"]["timeout"] if "timeout" in error["details"] else None,
            }
        )
    return errors
