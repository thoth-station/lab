import logging
import sys

from typing import Union, List, Dict, Any
from thoth.storages import DependencyMonkeyReportsStore

_LOGGER = logging.getLogger("thoth.lab.depedendency_monkey")

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def aggregate_dm_results_per_identifier(
    identifiers_inspection: List[str], limit_results: bool = False, max_batch_identifiers_ids: int = 5
) -> Union[dict, List[str]]:
    """Aggregate inspection batch ids and specification from dm documents stored in Ceph.

    :param inspection_identifier: list of identifier/s to filter inspection batch ids
    :param limit_results: reduce the number of inspection batch ids consideredto `max_batch_identifiers_ids` to test analysis
    :param max_batch_identifiers_ids: maximum number of inspection batch ids considered
    """
    dm_store = DependencyMonkeyReportsStore()
    dm_store.connect()

    dm_ids = list(dm_store.get_document_listing())

    _LOGGER.info("Number of DM reports identified is: %r" % len(dm_ids))

    dm_info_dict = {}
    inspection_batch_identifiers = []
    number_dm_ids = len(dm_ids)
    current_dm_counter = 1
    inspection_batch_counter = 0

    if limit_results:
        _LOGGER.info(f"Limiting results to {max_batch_identifiers_ids} to test functions!!")

    for n, ids in enumerate(dm_ids):
        document = dm_store.retrieve_document(ids)
        _LOGGER.info(f"Analysis n.{current_dm_counter}/{number_dm_ids}")
        report = document["result"].get("report")
        inspection_batch_ids_specifications = {}

        inspection_batch_ids_specifications, inspection_batch_identifiers, inspection_batch_counter = _extract_dm_responses_from_report(
            report=report,
            inspection_specification=inspection_batch_ids_specifications,
            inspection_batch_identifiers=inspection_batch_identifiers,
            identifiers=identifiers_inspection,
            inspection_batch_counter=inspection_batch_counter,
            max_ids=max_batch_identifiers_ids,
            limit_results=limit_results,
        )

        if inspection_batch_ids_specifications:
            _LOGGER.info(f"\nTot inspections batches identified: {len(inspection_batch_ids_specifications)}")
            dm_info_dict[ids] = {}
            dm_info_dict[ids] = inspection_batch_ids_specifications
        else:
            _LOGGER.info(f"No inspections batches identified")

        current_dm_counter += 1

        if limit_results:
            if inspection_batch_counter > max_batch_identifiers_ids:
                _LOGGER.info(f"\nTot inspections batch for the analysis: {len(inspection_batch_identifiers)}")
                return dm_info_dict, inspection_batch_identifiers

    _LOGGER.info(f"Tot inspections batch considered: {len(inspection_batch_identifiers)}")

    return dm_info_dict, inspection_batch_identifiers


def _extract_dm_responses_from_report(
    report: Dict[str, Any],
    inspection_specification: Dict[str, Any],
    inspection_batch_identifiers: List[str],
    identifiers: List[str],
    inspection_batch_counter: int,
    limit_results: bool = False,
    max_ids: int = 5,
) -> Dict[str, Any]:
    """Extract responses from Dependency Monkey reports."""
    if not report:
        return inspection_specification, inspection_batch_identifiers, inspection_batch_counter

    if limit_results:
        if inspection_batch_counter > max_ids:
            return inspection_specification, inspection_batch_identifiers, inspection_batch_counter

    responses = report.get("responses")
    inspection_specification, inspection_batch_identifiers, inspection_batch_counter = _extract_dm_product_from_responses(
        responses=responses,
        inspection_specification=inspection_specification,
        inspection_batch_identifiers=inspection_batch_identifiers,
        identifiers=identifiers,
        inspection_batch_counter=inspection_batch_counter,
        max_ids=max_ids,
        limit_results=limit_results,
    )

    return inspection_specification, inspection_batch_identifiers, inspection_batch_counter


def _extract_dm_product_from_responses(
    responses: List[Dict[str, Any]],
    inspection_specification: Dict[str, Any],
    inspection_batch_identifiers: List[str],
    identifiers: List[str],
    inspection_batch_counter: int,
    limit_results: bool = False,
    max_ids: int = 5,
) -> Dict[str, Any]:
    """Extract products per inspection id matching inspection identifier from Dependency Monkey reports."""
    if not responses:
        return inspection_specification, inspection_batch_identifiers, inspection_batch_counter

    for response in responses:
        for identifier in identifiers:
            if identifier in response["response"]:
                product = response["product"]

                inspection_specification[response["response"]] = {
                    "requirements": product["project"]["requirements"],
                    "requirements_locked": product["project"]["requirements_locked"],
                    "runtime_environment": product["project"]["runtime_environment"],
                }

                inspection_batch_identifiers.append(response["response"])
                inspection_batch_counter += 1

                if limit_results:
                    if inspection_batch_counter > max_ids:
                        return inspection_specification, inspection_batch_identifiers, inspection_batch_counter

    return inspection_specification, inspection_batch_identifiers, inspection_batch_counter
