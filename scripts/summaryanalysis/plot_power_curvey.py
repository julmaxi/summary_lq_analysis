import argparse

import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("files", nargs="+", type=lambda x: x.split(":"))

	args = parser.parse_args()

	for fname, label in args.files:
		result_df = pd.read_csv(fname, index_col=0, names=["annotator_count", "correct", "mismatch", "new"])

		print(result_df)

		scores = result_df.groupby("annotator_count").mean().sort_index()

		print(scores)

		plt.plot(scores.index.to_frame()["annotator_count"].to_numpy(), scores["correct"].to_numpy())

	plt.show()