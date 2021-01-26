from collections import Counter, defaultdict
import itertools as it

def get_annotator_groups(annotations):
    all_groups = defaultdict(list)
    for annotator, group in annotations.groupby("annotator").groups.items():
        docs = tuple(sorted(set(group.to_frame()["document"])))
        all_groups[docs].append(annotator)

    all_annotator_groups = list(map(tuple, all_groups.values()))
    return all_annotator_groups

