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

import argparse
import gzip
import logging
import sys
import json
import numpy as np
import scipy.stats as stats

from collections import namedtuple

import tqdm
from core import regiontools, common

def load_target_regions(fname):
    regions = []
    with open(fname, "r") as bed_file:
        for line in bed_file:
            chrom, start, end, *_ = line.split()
            start, end = int(start), int(end)
            region = regiontools.Region(chrom, start, end)
            regions.append(region)
    return regions


Parameters = namedtuple(
    "Parameters", ["manifest_path", "multisample_profile_path", "output_path", "target_region_path"]
)


def generate_table_with_anchor_counts(combined_counts):
    count_table = []
    for unit, rec in combined_counts.items():
        if "RegionsWithIrrAnchors" not in rec:
            continue

        for region, sample_counts in rec["RegionsWithIrrAnchors"].items():
            table_row = {"region": region, "unit": unit}
            table_row["sample_counts"] = sample_counts
            count_table.append(table_row)

    return count_table


def run(params):
    with open(params.multisample_profile_path, "r") as profile_file:
        multisample_profile = json.load(profile_file)
    count_table = generate_table_with_anchor_counts(multisample_profile["Counts"])
    logging.info("Loaded %i regions", len(count_table))

    logging.info("Normalizing counts")
    sample_stats = multisample_profile["Parameters"]
    common.depth_normalize_counts(sample_stats, count_table)


    if params.target_region_path:
        target_regions = load_target_regions(params.target_region_path)
        logging.info("Restricting analysis to %i regions", len(target_regions))
        count_table = common.filter_counts_by_region(count_table, target_regions)

    manifest = common.load_manifest(params.manifest_path)
    sample_status = common.extract_case_control_assignments(manifest)

    header = "contig\tstart\tend\tmotif\tthreshold\tsigma\tmu\tnormtest_stat\tnormtest_p\tchi2\tchi2p\twelchtstat\twelchpval\tmannwhitneystat\tmannwhitneyp\tcase_skew\tcase_inlier_skew\tcase_outlier_skew\tcontrol_skew\tcontrol_inlier_skew\tcontrol_outlier_skew\tcase_kurtosis\tcase_inlier_kurtosis\tcase_outlier_kurtosis\tcontrol_kurtosis\tcontrol_inlier_kurtosis\tcontrol_outlier_kurtosis\tdetected_cases\tdetected_controls\ttop_case_zscore\thigh_case_counts\tcase_counts\tcase_zscores\ttop_control_zscore\thigh_control_counts\tcontrol_counts\tcontrol_zscores"
    with open(params.output_path, "wt") as results_file:
        results_file.write(header)
        results_file.write("\n")
        for row in tqdm.tqdm(count_table):
            region_encoding = row["region"]
            if region_encoding == "unaligned":
                continue

            contig, coords = region_encoding.rsplit(":", 1)
            start, end = coords.split("-")
            start, end = int(start), int(end)

            threshold, sigma, mu, normtest_stat, normtest_p, chi2, chi2p, welchtstat, welchpval, mannwhitneystat, \
             mannwhitneyp, case_skew, case_inlier_skew, case_outlier_skew, control_skew, control_inlier_skew, \
             control_outlier_skew, case_kurtosis, case_inlier_kurtosis, case_outlier_kurtosis, control_kurtosis, \
             control_inlier_kurtosis, control_outlier_kurtosis, detected_cases, detected_controls, top_case_zscore, \
             cases_with_high_counts, case_counts, case_zscores, top_control_zscore, \
             controls_with_high_counts, control_counts, control_zscores = common.run_zscore_analysis(
                sample_status, row["sample_counts"]
            )

            if len(cases_with_high_counts) == 0:
                continue

            encoded_case_label_info = ",".join(
                "{}:{:.2f}".format(s, c) for s, c in cases_with_high_counts.items()
            )
            encoded_control_label_info = ",".join(
                "{}:{:.2f}".format(s, c) for s, c in controls_with_high_counts.items()
            )
            encoded_case_count_info = ",".join([str(x) for x in case_counts])
            encoded_case_zscore_info = ",".join([str(x) for x in case_zscores])
            encoded_control_count_info = ",".join([str(x) for x in control_counts])
            encoded_control_zscore_info = ",".join([str(x) for x in control_zscores])
            out_list = [contig,
                start,
                end,
                row["unit"],
                threshold,
                sigma,
                mu,
                normtest_stat, normtest_p, chi2, chi2p, welchtstat, welchpval, mannwhitneystat, mannwhitneyp, case_skew, case_inlier_skew, case_outlier_skew, control_skew, control_inlier_skew, control_outlier_skew, case_kurtosis, case_inlier_kurtosis, case_outlier_kurtosis, control_kurtosis, control_inlier_kurtosis, control_outlier_kurtosis,
                detected_cases,
                detected_controls,
                "{:.2f}".format(top_case_zscore),
                encoded_case_label_info,
                encoded_case_count_info,
                encoded_case_zscore_info,
                "{:.2f}".format(top_control_zscore),
                encoded_control_label_info,
                encoded_control_count_info,
                encoded_control_zscore_info]
            results_file.write("\t".join([str(x) for x in out_list]))
            results_file.write("\n")
    logging.info("Done")
