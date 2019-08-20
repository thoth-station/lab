# thoth-lab
# Copyright(C) 2018, 2019 Francesco Murdaca
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

logger = logging.getLogger("thoth.lab.inspection_report")

_INSPECTION_REPORT_FEATURES = {
    "hardware": ["platform", "processor", "ncpus"],
    "software_stack": ["index", "requirements_locked"],
    "base_image": [],
    "pi": ["script", "parameters"],
    "requests": ["build", "run"],
    "exit_codes": ["build", "run"],
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
    "job_exit_code": ["job__exit_code", "job_log__exit_code"],
}


def create_report(df_inspection_batch: pd.DataFrame) -> dict:
    """Create report describing the batch of inspection jobs for the different features."""
    report_results = {}
    for feature, sub_features in _INSPECTION_REPORT_FEATURES.items():
        report_results[feature] = {}
        print("\n=========================================================================")
        print(feature)
        print("\n=========================================================================")
        if sub_features:
            for sub_feature in sub_features:
                print("\n-------------------------------------------------------------------------")
                print(f"{feature} -> {sub_feature}")
                print("\n-------------------------------------------------------------------------")
                sub_feature_result = inspection.query_inspection_dataframe(
                    df_inspection_batch, groupby=_INSPECTION_JSON_DF_KEYS_FEATURES_MAPPING[sub_feature], exclude="node"
                )
                report_results[feature][sub_feature] = show_categories(sub_feature_result)
        else:
            feature_result = inspection.query_inspection_dataframe(
                df_inspection_batch, groupby=_INSPECTION_JSON_DF_KEYS_FEATURES_MAPPING[feature], exclude="node"
            )
            report_results[feature] = show_categories(sub_feature_result)

    return report_results


def create_feature_summary(df_inspection_batches_dict: dict, explanation: bool = False) -> dict:
    """Create summary of number of combinations per features."""
    results_features = _aggregate_results_per_feature(df_inspection_batches_dict)
    features_summary = {}
    explanation_summary = {}
    for feature, feature_results in results_features.items():
        explanation_summary[feature] = {}

        if not isinstance(feature_results, list):
            features_summary[feature] = {}

            for sub_feature, sub_feature_results in feature_results.items():
                explanation_summary[feature][sub_feature] = {}
                keys = [key for key in sub_feature_results[0].keys()]
                key_counts = {}

                for key in keys:

                    key_counts[key] = len(set([k[key] for k in results_features[feature][sub_feature]]))

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

                if key_counts[key] != 1:
                    explanation_summary[feature][key] = set([k[key] for k in results_features[feature]])

            features_summary[feature] = key_counts
    if explanation:
        return _visualize_differences_in_inspection_results(explanation_summary)

    return _visualize_summary(features_summary)


def _aggregate_results_per_feature(inspection_batches_reports_dict: dict) -> dict:
    """Aggregate results for all features across all batches."""
    results_per_feature_per_batch = {}

    for feature, sub_features in _INSPECTION_REPORT_FEATURES.items():
        for batch in inspection_batches_reports_dict.keys():

            if sub_features:
                if feature not in results_per_feature_per_batch.keys():
                    results_per_feature_per_batch[feature] = {}

                for sub_feature in sub_features:

                    if sub_feature not in results_per_feature_per_batch[feature].keys():
                        results_per_feature_per_batch[feature][sub_feature] = []

                    for k, v in inspection_batches_reports_dict[batch][feature][sub_feature].items():
                        results_per_feature_per_batch[feature][sub_feature].append(
                            inspection_batches_reports_dict[batch][feature][sub_feature][k]
                        )
            else:

                if feature not in results_per_feature_per_batch.keys():
                    results_per_feature_per_batch[feature] = []

                for k, v in inspection_batches_reports_dict[batch][feature].items():
                    results_per_feature_per_batch[feature].append(inspection_batches_reports_dict[batch][feature][k])

    return results_per_feature_per_batch


def _visualize_summary(summary_results: dict):
    """Visualize summary of results for all inspection batches (if there are any differences)."""
    for feature, feature_results in summary.items():
        print("\n===============================================================================")
        print(feature)
        print("\n===============================================================================")
        if len(feature_results) > 1:
            for sub_feature, sub_feature_results in feature_results.items():
                print("\n---------------------------------------------------------------------------")
                print(sub_feature)
                for key, count in sub_feature_results.items():
                    if count > 1:
                        print(f"{key}: {count}")
                    else:
                        print("All constant!")
        else:
            for key, count in feature_results.items():
                if count > 1:
                    print("\n===========================================================================================")
                    print(feature)
                    print("\n===========================================================================================")
                    print(f"{key}: {count}")
                else:
                    print("All constant!")


def _visualize_differences_in_inspection_results(summary_explained: dict):
    """Function to identify and visualize differences in inspection batches for the different features."""
    for feature, feature_results in summary_explained.items():
        print("\n=========================================================================")
        print(feature)
        print("\n=========================================================================")
        if summary_explained[feature]:
            for sub_feature, sub_feature_results in feature_results.items():
                if sub_feature_results:
                    print("\n-------------------------------------------------")
                    print(sub_feature)
                    print("\n-------------------------------------------------")
                    for key_sf in sub_feature_results.keys():
                        for value in sub_feature_results[key_sf]:
                            print()
                            print(f"{key_sf}: {value}")
                            for batch, batch_results in tot_reports.items():
                                for b, r in batch_results[feature][sub_feature].items():
                                    if r[key_sf] == value:
                                        print("Identifier:", batch)
        else:
            if feature_results:
                for key_f in feature_results.keys():
                    for value in sub_feature_results[key_f]:
                        print("=========================================================================")
                        print(f"\n{key_f}: {value}")
                        for batch, batch_results in tot_reports.items():
                            for b, r in batch_results[feature].items():
                                if r[key_f] == value:
                                    print("Identifier:", batch)
