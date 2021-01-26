import random
import itertools as it
from tqdm.auto import tqdm
import scipy.stats
from summaryanalysis.annotationutils import get_annotator_groups
import numpy as np


def compute_grouped_subsample_variance(annotations, crossed=False, score_name="coherence_score", limit=10000):
    groups = get_annotator_groups(annotations)

    annotations = annotations.sort_index()
    original_scores = annotations.groupby("system").mean().sort_index()[score_name].to_numpy()
    qualities = []
    annotation_costs = []

    for sample_size in tqdm(range(len(groups)), leave=False):
        sample_qualities = []

        for idx in range(limit):
            sampled_groups = random.sample(groups, sample_size + 1)
            if crossed:
                sampled_annotators = list(it.chain(*sampled_groups))
            else:
                sampled_annotators = [random.choice(g) for g in sampled_groups]

            sampled_annotations = annotations.loc[sampled_annotators]

            annotation_vec = sampled_annotations.groupby("system").mean().sort_index()[score_name].to_numpy()
            correlation, _ = scipy.stats.pearsonr(
                    annotation_vec,
                    original_scores)
            sample_qualities.append(correlation)

        annotation_cost = (sample_size + 1)
        if crossed:
            annotation_cost *= 3
        annotation_costs.append(annotation_cost)
        qualities.append(np.array(sample_qualities).mean())

    return annotation_costs, qualities


def compute_time_reliability_curve(annotations, times, score_key="coherence_score"):
    groups = get_annotator_groups(annotations)

    annotator_times = times.groupby("annotator").sum()

    original_scores = annotations.groupby("system").mean().sort_index()[score_key].to_numpy()

    all_scores = []
    all_times = []
    for sample_size in range(2, len(groups) - 1):
        combs = it.combinations(groups, sample_size)
        combs = list(combs)
        random.shuffle(combs)
        combs = combs[:500]

        for comb in tqdm(combs, leave=False):
            key = [a for b in comb for a in b]

            sample_scores = annotations.loc[key].groupby("system").mean().sort_index()[score_key].to_numpy()
            corr = scipy.stats.pearsonr(original_scores, sample_scores)[0]

            all_scores.append(corr)
            all_times.append(annotator_times.loc[key].sum())

    return all_scores, all_times
