from . import ordinal
import tempfile
import os
import subprocess
import tqdm
from collections import Counter
from multiprocessing import Pool
import argparse
import csv
import numpy as np



def regress_on_sample(args):
    num_blocks, distribution = args
    np.random.seed()
    sample = ordinal.MODELS[distribution].sample(ordinal.create_design(num_blocks, 5, 1))
    differences, p_values = run_regression(sample, nested=True)
    return differences, p_values


def run_experiment(num_blocks, num_iters, log_filename=None, distribution="likertD:multi_news:modified"):
    diffs_of_interest = [
        ("__REFERENCE__", "BART"),
        ("abssentrw", "onmt_pg")
    ]

    diffs_found = Counter()
    log_writer = None
    log_file = None
    if log_filename is not None:
        log_file = open(log_filename, "w")
        log_writer = csv.writer(log_file)
    with Pool() as pool:
        for differences, p_values in tqdm.tqdm(pool.imap_unordered(regress_on_sample, [(num_blocks, distribution)] * num_iters), total=num_iters):
            diffs_found.update(differences.intersection(diffs_of_interest))
            if log_writer:
                vals = []
                for diff in diffs_of_interest:
                    vals.append(p_values[tuple(sorted(diff))])
                log_writer.writerow(vals)

    if log_file is not None:
        log_file.close()

    for key, val in diffs_found.items():
        print(key, val/num_iters)


def run_regression(group_df, nested=False):
    mode = "crossed"
    if nested:
        mode = "nested"

    mode += ":none"

    handle, path = tempfile.mkstemp()
    f_temp = os.fdopen(handle, 'w')
    group_df.to_csv(path)
    f_temp.close()
    regression_result = subprocess.run(["Rscript", "scripts/r/analyse-ordinal.r", path, "score", mode], capture_output=True, encoding="utf8")
    os.remove(path)

    diffs = regression_result.stdout.split("\n")

    differences = set()
    p_values = {}

    for diff in diffs:
        if len(diff) == 0:
            continue
        pair, direction, p_value = diff.split("\t")
        p_value = float(p_value)

        sys_a, sys_b = pair.split(" - ")

        if direction == "-":
        	sys_a, sys_b = sys_b, sys_a

        if direction != "o":
            differences.add((sys_a, sys_b))

        p_values[sys_a, sys_b] = p_value

    return differences, p_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", dest="num_blocks", default=20, type=int)
    parser.add_argument("-i", dest="num_iters", default=1000, type=int)
    parser.add_argument("-l", dest="log_filename", default=None)
    parser.add_argument("-d", dest="distribution", default="likertD:multi_news:modified")

    args = parser.parse_args()

    run_experiment(args.num_blocks, args.num_iters, args.log_filename, args.distribution)

