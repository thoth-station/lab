# thoth-lab
# Copyright(C) 2018, 2019, 2020 Francesco Murdaca
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

"""Inspection report generation and visualization."""

import logging
import re

import pandas as pd

from typing import Any, Dict, List, Tuple, Union
from thoth.lab import inspection

logger = logging.getLogger("thoth.lab.inspection_report")

_INSPECTION_REPORT_FEATURES = {
    "hardware": ["platform", "processor", "ncpus"],
    "software_stack": ["index", "requirements_locked"],
    "base_image": [],
    "pi": ["script", "parameters"],
    "requests": ["build_requests", "run_requests"],
    "exit_codes": ["build_exit_code", "run_exit_code"],
}

_INSPECTION_JSON_DF_KEYS_FEATURES_MAPPING = {
    "platform": ["platform"],
    "processor": ["job_log__hwinfo__cpu__is", "job_log__hwinfo__cpu__has"],
    "ncpus": ["ncpus"],
    "index": ["index"],
    "requirements_locked": ["specification__python__requirements_locked__default"],
    "base_image": ["base"],
    "script": ["script", "script_sha256"],
    "parameters": ["name", "@parameters"],
    "build_requests": ["build__requests"],
    "run_requests": ["run__requests"],
    "build_exit_code": ["build__exit_code"],
    "run_exit_code": ["job__exit_code", "job_log__exit_code"],
}


def create_inspection_report(inspection_df: pd.DataFrame) -> dict:
    """Create report describing the batch of inspection jobs for the different features.

    :param inspection_df: data frame to analyze as returned by `process_inspection_results`
    """
    inspection_report = {}
    for feature, sub_features in _INSPECTION_REPORT_FEATURES.items():
        inspection_report[feature] = {}
        logger.info("\n=========================================================================")
        logger.info(feature)
        logger.info("=========================================================================")
        if sub_features:
            for sub_feature in sub_features:
                logger.info("-------------------------------------------------------------------------")
                logger.info(f"{feature} -> {sub_feature}")
                logger.info("-------------------------------------------------------------------------")
                sub_feature_result = inspection.query_inspection_dataframe(
                    inspection_df, groupby=_INSPECTION_JSON_DF_KEYS_FEATURES_MAPPING[sub_feature], exclude="node"
                )
                inspection_report[feature][sub_feature] = inspection.show_categories(sub_feature_result)
        else:
            feature_result = inspection.query_inspection_dataframe(
                inspection_df, groupby=_INSPECTION_JSON_DF_KEYS_FEATURES_MAPPING[feature], exclude="node"
            )
            inspection_report[feature] = inspection.show_categories(feature_result)

    return inspection_report


def create_inspection_reports(inspection_df_dict: dict) -> dict:
    """Create dictionary containing all reports for inspection batches selected.

    :param inspection_df_dict: dictionary with inspection_df per identifier as returned by process_inspection_results
    """
    identifier_list = list(inspection_df_dict.keys())

    inspection_report_dict = {}

    for identifier in identifier_list:
        inspection_report_dict[identifier] = create_inspection_report(inspection_df_dict[identifier])

    return inspection_report_dict


def create_feature_analysis_summary(inspection_report_dict: dict, explanation: bool = False):
    """Create summary of analysis of features across all inspection reports.

    :param inspection_report_dict: dictionary of inspection report as returned by create_inspection_report_dict
    :param explanation: flag to obtain more detailed results about differences in the inspection batches
    """
    results_features = _aggregate_results_per_feature(inspection_report_dict)
    features_summary = {}

    if explanation:
        explanation_summary = {}

    for feature, feature_results in results_features.items():
        if explanation:
            explanation_summary[feature] = {}

        if not isinstance(feature_results, list):
            features_summary[feature] = {}

            for sub_feature, sub_feature_results in feature_results.items():
                if explanation:
                    explanation_summary[feature][sub_feature] = {}
                keys = [key for key in sub_feature_results[0].keys()]
                key_counts = {}

                for key in keys:

                    key_counts[key] = len(set([k[key] for k in results_features[feature][sub_feature]]))
                    if explanation:
                        if key_counts[key] != 1:
                            explanation_summary[feature][sub_feature][key] = set(
                                [k[key] for k in results_features[feature][sub_feature]]
                            )

                features_summary[feature][sub_feature] = key_counts
        else:
            keys = [key for key in feature_results[0].keys()]
            key_counts = {}

            for key in keys:
                key_counts[key] = len(set([k[key] for k in results_features[feature]]))
                if explanation:
                    if key_counts[key] != 1:
                        explanation_summary[feature][key] = set([k[key] for k in results_features[feature]])

            features_summary[feature] = key_counts

    if explanation:
        return _visualize_differences_in_inspection_results(explanation_summary, inspection_report_dict)

    return _visualize_summary(features_summary)


def _aggregate_results_per_feature(inspection_report_dict: dict) -> dict:
    """Aggregate results for all features across all batches.

    :param inspection_report_dict: dictionary of inspection report as returned by create_inspection_report_dict
    """
    results_per_feature_per_batch = {}

    for feature, sub_features in _INSPECTION_REPORT_FEATURES.items():
        for batch in inspection_report_dict.keys():

            if sub_features:
                if feature not in results_per_feature_per_batch.keys():
                    results_per_feature_per_batch[feature] = {}

                for sub_feature in sub_features:

                    if sub_feature not in results_per_feature_per_batch[feature].keys():
                        results_per_feature_per_batch[feature][sub_feature] = []

                    for k, v in inspection_report_dict[batch][feature][sub_feature].items():
                        results_per_feature_per_batch[feature][sub_feature].append(
                            inspection_report_dict[batch][feature][sub_feature][k]
                        )
            else:

                if feature not in results_per_feature_per_batch.keys():
                    results_per_feature_per_batch[feature] = []

                for k, v in inspection_report_dict[batch][feature].items():
                    results_per_feature_per_batch[feature].append(inspection_report_dict[batch][feature][k])

    return results_per_feature_per_batch


def _visualize_summary(reports_summary: dict):
    """Visualize summary of results for all inspection batches (if there are any differences).

    :param reports_summary: summary of the reports analyzed per inspection identifier
    """
    for feature, feature_results in reports_summary.items():
        logger.info("===============================================================================")
        logger.info(feature)
        logger.info("===============================================================================")

        if len(feature_results) > 1:
            for sub_feature, sub_feature_results in feature_results.items():
                logger.info("---------------------------------------------------------------------------")
                logger.info(sub_feature)
                for key, count in sub_feature_results.items():
                    if count > 1:
                        logger.info(f"{key}: {count}")

        else:
            for key, count in feature_results.items():
                if count > 1:
                    logger.info(
                        "==========================================================================================="
                    )
                    logger.info(feature)
                    logger.info(
                        "==========================================================================================="
                    )
                    logger.info(f"{key}: {count}")


def _visualize_differences_in_inspection_results(detailed_reports_summary: dict, inspection_report_dict: dict):
    """Function to identify and visualize differences in inspection batches for the different features.

    :param detailed_reports_summary: detailed summary of the reports analyzed per inspection identifier
    :param inspection_report_dict: dictionary of inspection report as returned by create_inspection_report_dict
    """
    for feature, feature_results in detailed_reports_summary.items():
        logger.info("=========================================================================")
        logger.info(feature)
        logger.info("=========================================================================")

        if detailed_reports_summary[feature]:

            for sub_feature, sub_feature_results in feature_results.items():
                if sub_feature_results:
                    logger.info("-------------------------------------------------")
                    logger.info(sub_feature)
                    logger.info("-------------------------------------------------")
                    for key_sf in sub_feature_results.keys():

                        for value in sub_feature_results[key_sf]:
                            logger.info(f"{key_sf}: {value}")

                            for identifier, batch_results in inspection_report_dict.items():

                                for f_result in batch_results[feature][sub_feature].values():
                                    if f_result[key_sf] == value:
                                        logger.info("Identifier %r:", identifier)
        else:
            if feature_results:
                for key_f in feature_results.keys():

                    for value in sub_feature_results[key_f]:
                        logger.info("=========================================================================")
                        logger.info(f"{key_f}: {value}")

                        for identifier, batch_results in inspection_report_dict.items():

                            for f_result in batch_results[feature].values():
                                if f_result[key_f] == value:
                                    logger.info("Identifier %r:", identifier)
