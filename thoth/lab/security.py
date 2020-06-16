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

"""Security results processing and analysis."""

import logging
import json

import pandas as pd

from pathlib import Path

from plotly import graph_objs as go
from plotly.offline import iplot

# from thoth.storages import SIBanditResultsStore, SIClocResultsStore
from .common import aggregate_thoth_results
from typing import List, Optional, Tuple, Dict

from thoth.storages import SIBanditResultsStore
from thoth.storages import SIClocResultsStore

_LOGGER = logging.getLogger("thoth.lab.security")

logging.basicConfig(level=logging.INFO)


class SIBandit:
    """Class of methods used to analyze SI bandit analyzer results."""

    @staticmethod
    def aggregate_security_indicator_bandit_results(
        limit_results: bool = False,
        max_ids: int = 5,
        is_local: bool = True,
        security_indicator_bandit_repo_path: Path = Path("security/si-bandit"),
    ) -> list:
        """Aggregate si_bandit results from jsons stored in Ceph or locally from `si_bandit` repo.

        :param limit_results: reduce the number of si_bandit reports ids considered to `max_ids` to test analysis
        :param max_ids: maximum number of si_bandit reports ids considered
        :param is_local: flag to retreive the dataset locally or from S3 (credentials are required)
        :param si_bandit_repo_path: path to retrieve the si_bandit dataset locally and `is_local` is set to True
        """
        security_indicator_bandit_reports = aggregate_thoth_results(
            limit_results=limit_results,
            max_ids=max_ids,
            is_local=is_local,
            repo_path=security_indicator_bandit_repo_path,
            store=SIBanditResultsStore,
        )

        return security_indicator_bandit_reports

    @staticmethod
    def extract_data_from_si_bandit_metadata(report_metadata: dict) -> dict:
        """Extract data from report metadata."""
        extracted_metadata = {
            "datetime": report_metadata["datetime"],
            "analyzer": report_metadata["analyzer"],
            "analyzer_version": report_metadata["analyzer_version"],
            "document_id": report_metadata["document_id"],
            "package_name": report_metadata["arguments"]["si-bandit"]["package_name"],
            "package_version": report_metadata["arguments"]["si-bandit"]["package_version"],
            "package_index": report_metadata["arguments"]["si-bandit"]["package_index"],
        }

        return extracted_metadata

    def create_si_bandit_metadata_dataframe(self, si_bandit_report: dict) -> pd.DataFrame:
        """Create si-bandit report metadata dataframe."""
        metadata_si_bandit = self.extract_data_from_si_bandit_metadata(report_metadata=si_bandit_report["metadata"])
        metadata_df = pd.DataFrame([metadata_si_bandit])

        return metadata_df

    @staticmethod
    def extract_severity_confidence_info(
        si_bandit_report: dict, filters_files: Optional[List[str]] = None
    ) -> Tuple[List[dict], Dict[str, int]]:
        """Extract severity and confidence from result metrics."""
        extracted_info = []

        summary_files = {
            "number_of_analyzed_files": 0,
            "number_of_files_with_severities": 0,
            "number_of_filtered_files": 0,
        }

        if not filters_files:
            filters_files = []

        si_bandit_report_result_metrics_df = pd.DataFrame(si_bandit_report["result"]["metrics"])
        si_bandit_report_result_results_df = pd.DataFrame(si_bandit_report["result"]["results"])

        if "filename" not in si_bandit_report_result_results_df.columns.values:
            return extracted_info, summary_files

        for file in si_bandit_report_result_metrics_df.columns.values:
            # Filter tests/ file
            if file != "_totals" and not any(filter_ in file for filter_ in filters_files):

                analysis = {}
                analysis["name"] = file

                analysis["SEVERITY.LOW"] = {
                    "CONFIDENCE.LOW": 0,
                    "CONFIDENCE.MEDIUM": 0,
                    "CONFIDENCE.HIGH": 0,
                    "CONFIDENCE.UNDEFINED": 0,
                }
                analysis["SEVERITY.MEDIUM"] = {
                    "CONFIDENCE.LOW": 0,
                    "CONFIDENCE.MEDIUM": 0,
                    "CONFIDENCE.HIGH": 0,
                    "CONFIDENCE.UNDEFINED": 0,
                }
                analysis["SEVERITY.HIGH"] = {
                    "CONFIDENCE.LOW": 0,
                    "CONFIDENCE.MEDIUM": 0,
                    "CONFIDENCE.HIGH": 0,
                    "CONFIDENCE.UNDEFINED": 0,
                }

                subset_df = si_bandit_report_result_results_df[
                    si_bandit_report_result_results_df["filename"].values == file
                ]
                if subset_df.shape[0] > 0:
                    # check if there are severities for the file

                    for index, row in subset_df[["issue_confidence", "issue_severity"]].iterrows():
                        analysis[f"SEVERITY.{row['issue_confidence']}"][f"CONFIDENCE.{row['issue_severity']}"] += 1

                    summary_files["number_of_files_with_severities"] += 1

                summary_files["number_of_analyzed_files"] += 1

                extracted_info.append(analysis)

            elif file != "_totals" and any(filter_ in file for filter_ in filters_files):
                summary_files["number_of_filtered_files"] += 1

        return extracted_info, summary_files

    def create_security_confidence_dataframe(self, si_bandit_report: dict) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Create Security/Confidence dataframe for si-bandit report."""
        results_sec_conf, summary_files = self.extract_severity_confidence_info(
            si_bandit_report=si_bandit_report, filters_files=["tests/"]
        )

        summary_df = pd.DataFrame()

        if results_sec_conf:
            summary_df = pd.json_normalize(results_sec_conf, sep="__").set_index("name")
        else:
            summary_df = pd.json_normalize(results_sec_conf, sep="__")

        summary_df["_total_severity"] = summary_df.sum(axis=1)
        sec_conf_df = summary_df.transpose()
        sec_conf_df["_total"] = sec_conf_df.sum(axis=1)

        return sec_conf_df, summary_files

    @staticmethod
    def produce_si_bandit_report_summary_dataframe(
        metadata_df: pd.DataFrame, si_bandit_sec_conf_df: pd.DataFrame, summary_files: Dict[str, int]
    ) -> pd.DataFrame:
        """Create si-bandit report summary dataframe."""
        subset_df = pd.DataFrame([si_bandit_sec_conf_df["_total"].to_dict()])
        report_summary_df = pd.concat([metadata_df, subset_df], axis=1)
        report_summary_df["number_of_files_with_severities"] = int(summary_files["number_of_files_with_severities"])
        report_summary_df["number_of_analyzed_files"] = int(summary_files["number_of_analyzed_files"])
        report_summary_df["number_of_filtered_files"] = int(summary_files["number_of_filtered_files"])
        report_summary_df["number_of_files_total"] = int(summary_files["number_of_filtered_files"]) + int(
            summary_files["number_of_analyzed_files"]
        )
        report_summary_df["_total_severity"] = pd.to_numeric(report_summary_df["_total_severity"])

        return report_summary_df

    def create_si_bandit_final_dataframe(self, si_bandit_reports: List[dict]) -> pd.DataFrame:
        """Create final si-bandit dataframe."""
        counter = 1
        total_reports = len(si_bandit_reports)

        for si_bandit_report in si_bandit_reports:

            _LOGGER.info(f"Analyzing SI-bandit report: {counter}/{total_reports}")
            # Create metadata dataframe
            metadata_df = self.create_si_bandit_metadata_dataframe(si_bandit_report=si_bandit_report)
            _LOGGER.info(f"Analyzing package_name: {metadata_df['package_name'][0]}")
            _LOGGER.info(f"Analyzing package_version: {metadata_df['package_version'][0]}")
            _LOGGER.info(f"Analyzing package_index: {metadata_df['package_index'][0]}")

            # Create Security/Confidence dataframe
            security_confidence_df, summary_files = self.create_security_confidence_dataframe(
                si_bandit_report=si_bandit_report
            )

            si_bandit_report_summary_df = self.produce_si_bandit_report_summary_dataframe(
                metadata_df=metadata_df, si_bandit_sec_conf_df=security_confidence_df, summary_files=summary_files
            )

            if counter == 1:
                # Initialize columns in final dataframe
                final_df = pd.DataFrame(columns=si_bandit_report_summary_df.columns)

            # Add row, aka si-bandit report to final dataframe
            final_df.loc[counter] = si_bandit_report_summary_df.iloc[0]

            counter += 1

        return final_df

    @staticmethod
    def create_package_releases_vulnerabilities_trend(package_summary_df: pd.DataFrame):
        """Plot vulnerabilites trend for a package.

        :param package_summary_df dataframe for plot of vulnerabilites for a package from a certain index
        """
        package_name = package_summary_df["package_name"].values[0]
        package_index = package_summary_df["package_index"].values[0]

        X = package_summary_df["package_version"]
        data = []

        vulnerabilites_classes = [col for col in package_summary_df if col.startswith("SEVERITY.")]

        for vulnerability_class in vulnerabilites_classes:

            subset_df = package_summary_df[[vulnerability_class]]
            if subset_df.values.any():
                Z = [z[0] for z in subset_df.values]

                trace = go.Scatter(
                    x=X, y=Z, mode="markers+lines", marker=dict(size=4, opacity=0.8), name=f"{vulnerability_class}"
                )

                data.append(trace)

        for security_info in ["_total_severity"]:

            subset_df = package_summary_df[[security_info]]
            Z = [z[0] for z in subset_df.values]

            trace = go.Scatter(
                x=X, y=Z, mode="markers+lines", marker=dict(size=4, opacity=0.8), name=f"{security_info}"
            )

            data.append(trace)

        layout = go.Layout(
            title=f"SI analysis for {package_name} from {package_index} using SI-bandit",
            xaxis=dict(title="Releases"),
            yaxis=dict(title="Total number of vulnerabilities"),
            showlegend=True,
            legend=dict(orientation="h", y=-0.7, yanchor="top"),
        )
        fig = go.Figure(data=data, layout=layout)

        iplot(fig, filename="scatter-colorscale")

    @staticmethod
    def create_vulnerabilities_plot(security_df: pd.DataFrame):
        """Plot vulnerabilities for packages.

        :param security_df dataframe for plot of vulnerabilities for packages
        """
        packages = []
        vulnerabilites = {}
        total_severities = []

        severity_columns = [col for col in security_df if col.startswith("SEVERITY.")]

        for column in severity_columns:
            vulnerabilites[column] = []

        for index, row in security_df[
            ["package_name", "package_version", "package_index", "_total_severity"] + severity_columns
        ].iterrows():

            package_name = row["package_name"]
            package_version = row["package_version"]
            package_index = row["package_index"]

            packages.append(f"{package_name}-{package_version}-{package_index}")
            total_severities.append(row["_total_severity"])

            for column in severity_columns:
                vulnerabilites[column].append(row[column])

        data = []

        for vulnerability_class in vulnerabilites:
            trace = go.Scatter(
                x=packages,
                y=vulnerabilites[vulnerability_class],
                mode="markers+lines",
                marker=dict(size=4, opacity=0.8),
                name=f"{vulnerability_class}",
            )

            data.append(trace)

        infos = {"_total_severity": total_severities}
        for security_info in infos:
            trace = go.Scatter(
                x=packages,
                y=infos[security_info],
                mode="markers+lines",
                marker=dict(size=4, opacity=0.8),
                name=f"{security_info}",
            )

            data.append(trace)

        layout = go.Layout(
            title="SI analysis for Python packages using SI-bandit",
            xaxis=dict(title="{package_name-package_version-package_index}"),
            showlegend=True,
        )
        fig = go.Figure(data=data, layout=layout)

        iplot(fig, filename="scatter-colorscale")


class SICloc:
    """Class of methods used to analyze SI cloc analyzer results."""

    @staticmethod
    def aggregate_security_indicator_cloc_results(
        limit_results: bool = False,
        max_ids: int = 5,
        is_local: bool = True,
        security_indicator_cloc_repo_path: Path = Path("security/si-cloc"),
    ) -> list:
        """Aggregate si_cloc results from jsons stored in Ceph or locally from `si_cloc` repo.

        :param limit_results: reduce the number of si_cloc reports ids considered to `max_ids` to test analysis
        :param max_ids: maximum number of si_cloc reports ids considered
        :param is_local: flag to retreive the dataset locally or from S3 (credentials are required)
        :param si_cloc_repo_path: path to retrieve the si_cloc dataset locally and `is_local` is set to True
        """
        security_indicator_cloc_reports = aggregate_thoth_results(
            limit_results=limit_results,
            max_ids=max_ids,
            is_local=is_local,
            repo_path=security_indicator_cloc_repo_path,
            store=SIClocResultsStore,
        )

        return security_indicator_cloc_reports

    @staticmethod
    def extract_data_from_si_cloc_metadata(report_metadata: dict) -> dict:
        """Extract data from report metadata."""
        extracted_metadata = {
            "datetime": report_metadata["datetime"],
            "analyzer": report_metadata["analyzer"],
            "analyzer_version": report_metadata["analyzer_version"],
            "document_id": report_metadata["document_id"],
            "package_name": report_metadata["arguments"]["app.py"]["package_name"],
            "package_version": report_metadata["arguments"]["app.py"]["package_version"],
            "package_index": report_metadata["arguments"]["app.py"]["package_index"],
        }

        return extracted_metadata

    def create_si_cloc_metadata_dataframe(self, si_cloc_report: dict) -> pd.DataFrame:
        """Create si-cloc report metadata dataframe."""
        metadata_si_cloc = self.extract_data_from_si_cloc_metadata(report_metadata=si_cloc_report["metadata"])
        metadata_df = pd.DataFrame([metadata_si_cloc])

        return metadata_df

    def create_cloc_results_dataframe(self, si_cloc_report: dict) -> pd.DataFrame:
        """Create si-cloc report results dataframe."""
        results = {k: v for k, v in si_cloc_report["result"].items() if k != "header"}
        results["SUM"]["n_lines"] = si_cloc_report["result"]["header"]["n_lines"]
        results_df = pd.json_normalize(results)

        return results_df

    @staticmethod
    def produce_si_cloc_report_summary_dataframe(
        metadata_df: pd.DataFrame, cloc_results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create si-cloc report summary dataframe."""
        report_summary_df = pd.concat([metadata_df, cloc_results_df], axis=1)

        return report_summary_df

    def create_si_cloc_final_dataframe(self, si_cloc_reports: list) -> pd.DataFrame:
        """Create final si-cloc dataframe."""
        counter = 1
        total_reports = len(si_cloc_reports)

        final_df = pd.DataFrame()

        for si_cloc_report in si_cloc_reports:

            _LOGGER.info(f"Analyzing SI-cloc report: {counter}/{total_reports}")

            # Create metadata dataframe
            metadata_df = self.create_si_cloc_metadata_dataframe(si_cloc_report)
            _LOGGER.info(f"Analyzing package_name: {metadata_df['package_name'][0]}")
            _LOGGER.info(f"Analyzing package_version: {metadata_df['package_version'][0]}")
            _LOGGER.info(f"Analyzing package_index: {metadata_df['package_index'][0]}")

            # Create Security/Confidence dataframe
            cloc_results_df = self.create_cloc_results_dataframe(si_cloc_report=si_cloc_report)

            si_cloc_report_summary_df = self.produce_si_cloc_report_summary_dataframe(
                metadata_df=metadata_df, cloc_results_df=cloc_results_df
            )

            final_df = pd.concat([final_df, si_cloc_report_summary_df], axis=0)

            counter += 1

        return final_df
