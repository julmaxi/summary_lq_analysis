import random
import itertools as it
from collections import defaultdict


def get_annotator_groups(annotations):
    all_groups = defaultdict(list)
    for annotator, group in annotations.groupby("annotator").groups.items():
        docs = tuple(sorted(set(group.to_frame()["document"])))
        all_groups[docs].append(annotator)
    
    all_annotator_groups = list(map(tuple, all_groups.values()))
    return all_annotator_groups


def generate_samples(annotations, size, nested=False):
	groups = get_annotator_groups(annotations)

	combinations = list(it.combinations(groups, size))

	for group_combo in combinations:
		if nested:
			group_combo = [random.choice(g) for g in group_combo]
		else:
			group_combo = [m for g in group_combo for m in g]

		yield annotations.loc[group_combo], len(group_combo)