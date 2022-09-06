#
# ExpansionHunter Denovo
# Copyright 2016-2019 Illumina, Inc.
# All rights reserved.
#
# Author: Egor Dolzhenko <edolzhenko@illumina.com>,
#         Michael Eberle <meberle@illumina.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

import collections
import logging
import json
import scipy.stats as stats
import numpy as np
from scipy.stats import shapiro, chi2_contingency, skew, kurtosis, ttest_ind, mannwhitneyu

from . import regiontools

from .wilcoxontest import wilcoxon_rank_sum_test


def init_logger():
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)


def load_manifest(manifest_path):
    """Extract sample information from a manifest file.

    """

    # pylint: disable=I0011,C0103
    Sample = collections.namedtuple("Sample", "id status path")

    samples = []
    with open(manifest_path, "r") as manifest_file:
        for line in manifest_file:
            sample_id, status, path = line.split()

            if status not in ["case", "control"]:
                message = (
                    'Sample status must be either "case" or "control";'
                    ' instead got "{}"'
                )
                raise Exception(message.format(status))

            sample = Sample(id=sample_id, status=status, path=path)
            samples.append(sample)
    return samples


def filter_counts_by_magnitude(count_table, count_cutoff):
    filtered_count_table = []
    for row in count_table:
        max_count = max(count for _, count in row["sample_counts"].items())
        if max_count >= count_cutoff:
            filtered_count_table.append(row)

    return filtered_count_table


def filter_counts_by_region(count_table, target_regions):
    filtered_count_table = []
    for row in count_table:
        region_encoding = row["region"]
        chrom, coords = region_encoding.rsplit(":", 1)
        start, end = coords.split("-")
        start, end = int(start), int(end)
        region = regiontools.Region(chrom, start, end)
        overlaps_target_region = any(
            regiontools.compute_distance(region, target) == 0
            for target in target_regions
        )

        if overlaps_target_region:
            filtered_count_table.append(row)

    return filtered_count_table


def extract_case_control_assignments(samples):
    sample_status = {}
    for sample in samples:
        sample_status[sample.id] = sample.status
    return sample_status


def test_samples(test_params, sample_status, sample_counts):
    control_samples = [
        sample for sample, status in sample_status.items() if status == "control"
    ]
    case_samples = [
        sample for sample, status in sample_status.items() if status == "case"
    ]

    control_counts = [
        sample_counts[s] if s in sample_counts else 0 for s in control_samples
    ]

    case_counts = [sample_counts[s] if s in sample_counts else 0 for s in case_samples]

    pvalue = wilcoxon_rank_sum_test(test_params, case_counts, control_counts)

    return pvalue


def compare_counts(test_params, sample_status, count_table):
    for row in count_table:
        # Generate counts before testing
        pvalue = test_samples(test_params, sample_status, row["sample_counts"])
        row["pvalue"] = pvalue


def correct_pvalues(count_table):
    num_tests = len(count_table)
    for row in count_table:
        row["bonf_pvalue"] = min(row["pvalue"] * num_tests, 1.0)


def normalize_count(sample_depth, count, target_depth=40):
    return target_depth * count / sample_depth


def depth_normalize_counts(sample_stats, count_table):
    depths = sample_stats["Depths"]
    for row in count_table:
        row["sample_counts"] = {
            s: normalize_count(depths[s], c) for s, c in row["sample_counts"].items()
        }


def generate_table_with_irr_pair_counts(combined_counts):
    count_table = []
    for unit, rec in combined_counts.items():
        if "IrrPairCounts" not in rec:
            continue

        sample_counts = rec["IrrPairCounts"]
        table_row = {"unit": unit, "sample_counts": sample_counts}
        count_table.append(table_row)

    return count_table


def resample_quantiles(counts, num_resamples, target_quantile_value):
    resamples = np.random.choice(counts, len(counts) * num_resamples)
    resamples = np.split(resamples, num_resamples)

    resampled_quantiles = []
    for resample in resamples:
        quantile = np.quantile(resample, target_quantile_value)
        resampled_quantiles.append(quantile)

    return resampled_quantiles


def run_zscore_analysis(sample_status, sample_counts):
    ## 20220826: LF rewrite
    # Change: estimate quantiles in the control population _only_:
    control_counts = [sample_counts.get(sample, 0) for sample, status in sample_status.items() if status == "control"]
    quantiles = resample_quantiles(control_counts, 100, 0.95)
    (mu, sigma) = stats.norm.fit(quantiles)
    sigma = max(sigma, 1)
    shapiro_result = shapiro(quantiles)
    shapiro_w = shapiro_result.statistic
    shapiro_p = shapiro_result.pvalue
    case_counts = {
        sample: sample_counts.get(sample, 0)
        for sample, status in sample_status.items()
        if status == "case"
    }
    control_counts = {
        sample: sample_counts.get(sample, 0)
        for sample, status in sample_status.items()
        if status == "control"
    }
    assert len(case_counts) >= 1, "Manifest must contain at least one case"
    assert len(control_counts) >= 1, "Manifest must contain at least one control"
    # Change: record zscores for cases and controls in the control distribution:
    case_zscores = list()
    cases_with_high_counts = {}
    top_case_zscore = 0
    case_inliers = list()
    case_outliers = list()
    case_count_list = list()
    for sample, count in case_counts.items():
        zscore = (count - mu) / sigma
        if zscore > 1.0:
            case_outliers.append(count)
            cases_with_high_counts[sample] = count
            top_case_zscore = max(top_case_zscore, zscore)
        else:
            case_inliers.append(count)
        case_count_list.append(count)
        case_zscores.append(zscore)
    control_zscores = list()
    control_inliers = list()
    controls_with_high_counts = {}
    top_control_zscore = 0
    control_count_list = list()
    control_outliers = list()
    for sample, count in control_counts.items():
        zscore = (count - mu) / sigma
        if zscore > 1.0:
            control_outliers.append(count)
            controls_with_high_counts[sample] = count
            top_control_zscore = max(top_control_zscore, zscore)
        else:
            control_inliers.append(count)
        control_count_list.append(count)
        control_zscores.append(zscore)
    case_outliers = np.array(case_outliers)
    case_inliers = np.array(case_inliers)
    control_outliers = np.array(control_outliers)
    control_inliers = np.array(control_inliers)
    chi2, chi2p, dof, ex = chi2_contingency([[len(case_outliers), len(control_outliers)], [len(case_inliers), len(control_inliers)]])
    if len(case_outliers) != 0 and len(control_outliers) != 0:
        case_skew = skew(case_count_list)
        case_kurtosis = kurtosis(case_count_list)
        case_outlier_kurtosis = kurtosis(case_outliers)
        case_outlier_skew = skew(case_outliers)
        case_inlier_kurtosis = kurtosis(case_inliers)
        case_inlier_skew = skew(case_inliers)
        control_kurtosis = kurtosis(control_count_list)
        control_skew = skew(control_count_list)
        control_outlier_kurtosis = kurtosis(control_outliers)
        control_outlier_skew = skew(control_outliers)
        control_inlier_kurtosis = kurtosis(control_inliers)
        control_inlier_skew = skew(control_inliers)
        welcht_test = ttest_ind(case_outliers, control_outliers, equal_var=False, alternative="greater")
        welchtstat = welcht_test.statistic
        welchpval = welcht_test.pvalue
        mwu_test = mannwhitneyu(case_outliers, control_outliers, alternative="greater")
        mannwhitneystat = mwu_test.statistic
        mannwhitneyp = mwu_test.pvalue
    else:
        case_skew = np.nan
        case_kurtosis = np.nan
        case_outlier_kurtosis = np.nan
        case_outlier_skew = np.nan
        case_inlier_kurtosis = np.nan
        case_inlier_skew = np.nan
        control_kurtosis = np.nan
        control_skew = np.nan
        control_outlier_kurtosis = np.nan
        control_outlier_skew = np.nan
        control_inlier_kurtosis = np.nan
        control_inlier_skew = np.nan
        welchtstat = np.nan
        welchpval = np.nan
        mannwhitneystat = np.nan
        mannwhitneyp = np.nan
    detected_cases = max(0,len(list(filter(None, list(case_counts.values())))))
    detected_controls = max(0,len(list(filter(None, list(control_counts.values())))))
    return ((mu+sigma), sigma, mu, shapiro_w, shapiro_p, chi2, chi2p, welchtstat, welchpval, mannwhitneystat, mannwhitneyp, case_skew, case_inlier_skew, case_outlier_skew, control_skew, control_inlier_skew, control_outlier_skew, case_kurtosis, case_inlier_kurtosis, case_outlier_kurtosis, control_kurtosis, control_inlier_kurtosis, control_outlier_kurtosis, detected_cases, detected_controls, top_case_zscore,
            cases_with_high_counts, list(case_counts.values()), case_zscores, top_control_zscore,
            controls_with_high_counts, list(control_counts.values()), control_zscores)
