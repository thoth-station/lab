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

"""Dependency Monkey results processing and analysis."""

import logging
import sys

from typing import Union, List, Dict, Any, Tuple
from thoth.storages import DependencyMonkeyReportsStore

_LOGGER = logging.getLogger("thoth.lab.depedendency_monkey")

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def aggregate_dm_results_per_identifier(
    identifiers_inspection: List[str], limit_results: bool = False, max_batch_identifiers_ids: int = 5
) -> Union[dict, List[str]]:
    """Aggregate inspection batch ids and specification from dm documents stored in Ceph.

    :param inspection_identifier: list of identifier/s to filter inspection batch ids
    :param limit_results: limit inspection batch ids considered to `max_batch_identifiers_ids` to test analysis
    :param max_batch_identifiers_ids: maximum number of inspection batch ids considered
    """
    dm_store = DependencyMonkeyReportsStore()
    dm_store.connect()

    dm_ids = list(dm_store.get_document_listing())

    _LOGGER.info("Number of DM reports identified is: %r" % len(dm_ids))

    dm_info_dict = {}
    i_batch_identifiers = []
    number_dm_ids = len(dm_ids)
    i_batch_counter = 0

    if limit_results:
        _LOGGER.info(f"Limiting results to {max_batch_identifiers_ids} to test functions!!")

    for current_dm_counter, ids in enumerate(dm_ids):
        document = dm_store.retrieve_document(ids)
        _LOGGER.info(f"Analysis n.{current_dm_counter + 1}/{number_dm_ids}")
        report = document["result"].get("report")
        i_batch_ids_specifications = {}

        i_batch_ids_specifications, i_batch_identifiers, i_batch_counter = _extract_dm_responses_from_report(
            report=report,
            inspection_specifications=i_batch_ids_specifications,
            i_batch_identifiers=i_batch_identifiers,
            identifiers=identifiers_inspection,
            i_batch_counter=i_batch_counter,
            max_ids=max_batch_identifiers_ids,
            limit_results=limit_results,
        )

        if i_batch_ids_specifications:
            _LOGGER.info(f"\nTot inspections batches identified: {len(i_batch_ids_specifications)}")
            dm_info_dict[ids] = {}
            dm_info_dict[ids] = i_batch_ids_specifications
        else:
            _LOGGER.info(f"No inspections batches identified")

        if limit_results:
            if i_batch_counter > max_batch_identifiers_ids:
                _LOGGER.info(f"\nTot inspections batch for the analysis: {len(i_batch_identifiers)}")
                return dm_info_dict, i_batch_identifiers

    _LOGGER.info(f"Tot inspections batch considered: {len(i_batch_identifiers)}")

    return dm_info_dict, i_batch_identifiers


def _extract_dm_responses_from_report(
    report: Dict[str, Any],
    inspection_specifications: Dict[str, Any],
    i_batch_identifiers: List[str],
    identifiers: List[str],
    i_batch_counter: int,
    limit_results: bool = False,
    max_ids: int = 5,
) -> Union[dict, List[str], int]:
    """Extract responses from Dependency Monkey reports."""
    if not report:
        return inspection_specifications, i_batch_identifiers, i_batch_counter

    if limit_results:
        if i_batch_counter > max_ids:
            return inspection_specifications, i_batch_identifiers, i_batch_counter

    responses = report.get("responses")
    inspection_specifications, i_batch_identifiers, i_batch_counter = _extract_dm_product_from_responses(
        responses=responses,
        inspection_specifications=inspection_specifications,
        inspection_batch_identifiers=i_batch_identifiers,
        identifiers=identifiers,
        inspection_batch_counter=i_batch_counter,
        max_ids=max_ids,
        limit_results=limit_results,
    )

    return inspection_specifications, i_batch_identifiers, i_batch_counter


def _extract_dm_product_from_responses(
    responses: List[Dict[str, Any]],
    inspection_specifications: Dict[str, Any],
    inspection_batch_identifiers: List[str],
    identifiers: List[str],
    inspection_batch_counter: int,
    limit_results: bool = False,
    max_ids: int = 5,
) -> Union[dict, List[str], int]:
    """Extract products per inspection id matching inspection identifier from Dependency Monkey reports."""
    if not responses:
        return inspection_specifications, inspection_batch_identifiers, inspection_batch_counter

    for response in responses:
        for identifier in identifiers:
            if identifier in response["response"]:
                product = response["product"]

                inspection_specifications[response["response"]] = {
                    "requirements": product["project"]["requirements"],
                    "requirements_locked": product["project"]["requirements_locked"],
                    "runtime_environment": product["project"]["runtime_environment"],
                }

                inspection_batch_identifiers.append(response["response"])
                inspection_batch_counter += 1

                if limit_results:
                    if inspection_batch_counter > max_ids:
                        return inspection_specifications, inspection_batch_identifiers, inspection_batch_counter

    return inspection_specifications, inspection_batch_identifiers, inspection_batch_counter
