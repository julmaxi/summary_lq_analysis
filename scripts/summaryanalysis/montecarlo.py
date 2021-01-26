import pandas as pd
import scipy.stats

import summaryanalysis.ordinal as ordinal
from summaryanalysis.art import paired_approximate_randomization_test
from summaryanalysis.annotationutils import get_annotator_groups
from pathlib import Path
from tqdm.auto import tqdm
import re
import itertools as it
from collections import defaultdict, Counter


def get_model_type1_error_rates(model, blocks):
    model = model.copy()
    model.zero_coefficients()

    test_type_1_error_rates = defaultdict(list)

    tests = {
        "ttest": lambda x, y: scipy.stats.ttest_rel(x, y).pvalue,
        "art": paired_approximate_randomization_test
    }

    for n_blocks, n_docs in tqdm(blocks):
        success_counts = Counter()
        n_comb = 0

        for _ in tqdm(range(1000)):
            results = model.sample(ordinal.create_design(n_blocks, n_docs, 3))

            aggregated_results = results.groupby(["system", "document"]).mean()

            pairs = list(it.combinations(results.index.unique("system"), 2))

            for sys_1, sys_2 in pairs:
                sys_1_results_no_agg = results.xs(sys_1, level="system").to_numpy()
                sys_1_results_agg = aggregated_results.xs(sys_1, level="system").to_numpy()

                sys_2_results_no_agg = results.xs(sys_2, level="system").to_numpy()
                sys_2_results_agg = aggregated_results.xs(sys_2, level="system").to_numpy()

                for name, test in tests.items():
                    p_value_no_agg = test(sys_1_results_no_agg, sys_2_results_no_agg)
                    p_value_agg = test(sys_1_results_agg, sys_2_results_agg)

                    if p_value_no_agg < 0.05:
                        success_counts[name + "_no_agg"] += 1
                    if p_value_agg < 0.05:
                        success_counts[name + "_agg"] += 1

                n_comb += 1

        for key, n_succ in success_counts.items():
            test_type_1_error_rates[key].append(n_succ / n_comb)

    return test_type_1_error_rates


def read_obspower_files(path, pattern, filter_func=lambda *args: True):
    result_index = []
    result_entries = []

    for csv_path in Path(path).glob(pattern):
        n_blocks, n_docs, n_annotators = map(int, re.match(".*?(\d+)_(\d+)_(\d+).csv", csv_path.name).groups())
        if not filter_func(csv_path, n_blocks, n_docs, n_annotators):
            continue

        total_annotation_count = n_blocks * n_annotators * n_docs
        total_annotator_count = n_blocks * n_annotators

        data = pd.read_csv(csv_path, header=0, names=["better", "worse", "p_value"], index_col=[0, 1])

        result_entries.append(data)
        result_index.append((n_annotators, total_annotation_count, total_annotator_count))

    df = pd.concat(result_entries, names=["annotators", "effort", "total_annotators"], keys=result_index)
    return df


def filter_wrong_rankings(model, df):
    coefficients = dict(zip(model.systems, model.coefficients))
    correct_rankings = set((x, y) for ((x, cx), (y, cy)) in it.product(coefficients.items(), coefficients.items()) if cx > cy)
    mask = [t in correct_rankings for t in zip(df.index.to_frame()["better"], df.index.to_frame()["worse"])]
    new_df = df.copy()
    new_df["p_value"] = new_df["p_value"].where(mask, 1.0)
    return new_df


def add_grouping_column(df):
    group_df_entries = []

    for idx, group in enumerate(get_annotator_groups(df)):
        for member in group:
            group_df_entries.append({"annotator": member, "group": idx})

    grouping_df = pd.DataFrame.from_records(group_df_entries, index="annotator")
    df = df.join(grouping_df)
    df = df.set_index(['group'], append=True)
    return df


def get_art_pvals(model, design):
    result = []
    for _ in range(100):
        sample = model.sample(design)
        sample = add_grouping_column(sample)
        sample = sample.groupby(["system", "group"]).mean()

        for sys_1, sys_2 in it.combinations(sample.index.unique("system"), 2):
            sample_1 = sample.xs(sys_1, level="system").to_numpy()
            sample_2 = sample.xs(sys_2, level="system").to_numpy()
            p_val = paired_approximate_randomization_test(sample_1, sample_2)

            if sample_2.mean() > sample_1.mean():
                sys_1, sys_2 = sys_2, sys_1


            result.append({"better": sys_1, "worse": sys_2, "p_value": p_val})

    return pd.DataFrame.from_records(result, index=["better", "worse"])


def run_art_experiment(model, annotator_count, block_counts):
    all_pvals = []
    all_keys = []

    for n_annotators in (1, annotator_count):
        for n_blocks in block_counts:
            if n_annotators == 1:
                n_blocks *= annotator_count

            all_pvals.append(get_art_pvals(model, ordinal.create_design(n_blocks, 5, n_annotators)))
            all_keys.append((n_annotators, n_blocks, 5))

    df = pd.concat(all_pvals, keys=all_keys, names=["annotators", "blocks", "documents"])
    df = df.reset_index()
    df["effort"] = df["annotators"] * df["blocks"] * df["documents"]
    df["total_annotators"] = df["annotators"] * df["blocks"]
    return df.set_index(["annotators", "effort", "total_annotators", "better", "worse"], drop=True)


def run_art_experiment_fixed_budget(model, budget, annotator_count, block_counts):
    result = []
    all_keys = []
    for n_blocks in block_counts:
        pvals = get_art_pvals(model, ordinal.create_design(n_blocks, budget // n_blocks, annotator_count))
        result.append(pvals)
        all_keys.append((n_blocks, budget * annotator_count, n_blocks * annotator_count))

    return pd.concat(result, keys=all_keys, names=["annotators", "effort", "total_annotators"])


