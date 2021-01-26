import argparse

import pandas as pd
import numpy as np

from . import ordinal
from . import power

from multiprocessing import Pool
import itertools as it

from concurrent.futures import ThreadPoolExecutor

import os


def regress_on_sample(vals):
    model, design, nested = vals
    np.random.seed()
    sample_df = model.sample(design)
    diffs, p_values = power.run_regression(sample_df, nested=nested)

    return diffs, p_values


def test_design_power(model, design, nested=False, num_iters=100):
    results = []
    index = []
    idx = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for diffs, p_values in executor.map(regress_on_sample, [(model, design, nested)] * num_iters):
    #with Pool() as pool:
    #    for diffs, p_values in pool.imap_unordered(regress_on_sample, [(model, design, nested)] * num_iters):
            idx += 1
            print(f"{idx}/{num_iters}")
            for (left_system, right_system), p_val in p_values.items():
                results.append(p_val)
                index.append((left_system, right_system))

            if len(p_values) == 0:
                for sys_1, sys_2 in it.combinations(model.systems, 2):
                    results.append(1.0)
                    index.append((sys_1, sys_2))

    df = pd.DataFrame.from_dict({"p_value": results})
    df.index = pd.MultiIndex.from_tuples(index)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file")
    parser.add_argument("out_file")

    parser.add_argument("-b", dest="num_blocks", default=20, type=int)
    parser.add_argument("-d", dest="num_docs", default=5, type=int)
    parser.add_argument("-a", dest="num_annotators", default=3, type=int)
    parser.add_argument("-z", dest="zero_coefficients", default=False, action="store_true")
    parser.add_argument("-n", dest="condition_nested", default=False, action="store_true")

    args = parser.parse_args()

    model = ordinal.OrdinalModel.from_file(args.model_file)

    if args.zero_coefficients:
        model.zero_coefficients()

    analysis_result = test_design_power(model, ordinal.create_design(args.num_blocks, args.num_docs, args.num_annotators), nested=args.num_annotators == 1)
    analysis_result.to_csv(args.out_file)


def old():
    crossed_regression_results = []

    for block_count in range(1, 21):
        analysis_result = test_design_power(model, ordinal.create_design(block_count, 5, 3), nested=False)
        crossed_regression_results.append(analysis_result)

    result_df_crossed = pd.concat(crossed_regression_results, keys=[str(s * 3) for s in range(1, 21)], names=["annotator_count"])

    nested_regression_results = []

    for block_count in range(3, 61):
        analysis_result = test_design_power(model, ordinal.create_design(block_count, 5, 1), nested=True)
        nested_regression_results.append(analysis_result)

    result_df_nested = pd.concat(nested_regression_results, keys=[str(s) for s in range(3, 61)], names=["annotator_count"])

    result_df = pd.concat((result_df_crossed, result_df_nested), keys=["crossed", "nested"], names=["design"])

    print(result_df.groupby(["design", "block_count"]).mean())

    result_df.to_csv(args.out_file)
