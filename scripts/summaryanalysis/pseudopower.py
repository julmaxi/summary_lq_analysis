import argparse
import tempfile
import subprocess
import os
import csv

import pandas as pd
import tqdm

from .sample import generate_samples, get_annotator_groups


def take(iterable, n):
	for idx, item in enumerate(iterable):
		if idx == n:
			break
		else:
			yield item


def run_regression(group_df, nested=False):
	mode = "crossed"
	if nested:
		mode = "nested"

	handle, path = tempfile.mkstemp()
	f_temp = os.fdopen(handle, 'w')
	group_df.to_csv(path)
	f_temp.close()
	regression_result = subprocess.run(["Rscript", "analyse-ordinal.r", path, "coherence_score", mode], capture_output=True, encoding="utf8")
	os.remove(path)

	diffs = regression_result.stdout.split("\n")

	differences = set()

	for diff in diffs:
		if len(diff) == 0:
			continue
		pair, direction = diff.split("\t")
		sys_a, sys_b = pair.split(" - ")

		if direction == "-":
			sys_a, sys_b = sys_b, sys_a

		differences.add((sys_a, sys_b))

	return differences



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("annotation_file")
	parser.add_argument("out_file")
	parser.add_argument("-n", dest="nested", action="store_true", default=False)

	args = parser.parse_args()

	annotations = pd.read_csv(args.annotation_file, index_col=[0, 1, 2])

	results_log = open("regression.log", "w")
	result_file = open(args.out_file, "w")
	result_writer = csv.writer(result_file)

	base_differences = run_regression(annotations)

	contradictory_differences = set((b, a) for a, b in base_differences)

	for group_size in tqdm.tqdm(range(1, len(get_annotator_groups(annotations)))):
		num_detected = 0
		num_contradictions = 0
		num_new = 0
		for idx, (group_df, num_annotators) in enumerate(take(generate_samples(annotations, group_size, nested=args.nested), 10)):

			detected_differences = run_regression(group_df, nested=args.nested)
			results_log.write(f"#{group_size} {num_annotators} {idx}\n")
			for diff in sorted(detected_differences):
				results_log.write("\t".join(diff))
				results_log.write("\n")
			results_log.write("\n")

			agrees = base_differences.intersection(detected_differences)
			contradictions = contradictory_differences.intersection(detected_differences)
			new_diffs = detected_differences.difference(detected_differences).difference(contradictory_differences)

			num_detected += len(agrees)
			num_contradictions += len(contradictions)
			num_new += len(new_diffs)

			result_writer.writerow((num_annotators, len(agrees) / len(base_differences), len(contradictions) / len(contradictory_differences), len(new_diffs)))

		print(group_size, num_detected / (len(base_differences) * 10), num_contradictions, num_new)

	result_file.close()
	results_log.close()
