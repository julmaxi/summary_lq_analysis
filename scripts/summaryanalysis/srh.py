from collections import defaultdict
import itertools as it
import random
import scipy.stats
import numpy as np

def get_annotator_groups(annotations):
    all_groups = defaultdict(list)
    for annotator, group in annotations.groupby("annotator").groups.items():
        docs = tuple(sorted(set(group.to_frame()["document"])))
        all_groups[docs].append(annotator)

    all_annotator_groups = list(map(tuple, all_groups.values()))
    return all_annotator_groups

def compute_correlations_from_selection_mask(annotations, select_mask, score_name):
    scores_1 = annotations.loc[~select_mask][score_name]
    scores_2 = annotations.loc[select_mask][score_name]
    #scores_1 = scores_1[scores_1 != 0]
    #scores_2 = scores_2[scores_2 != 0]

    system_scores_1 = scores_1.groupby("system").mean().sort_index().to_numpy()
    system_scores_2 = scores_2.groupby("system").mean().sort_index().to_numpy()

    corr,_ = scipy.stats.pearsonr(system_scores_1, system_scores_2)

    sq_err = np.absolute((system_scores_1 - system_scores_2)).mean()
    return corr, sq_err


def compute_annotator_srh_raw(annotations, limit=1000, score_names=("coherence_score", "pronoun_score", "noun_phrase_score", "repetition_score")):
    all_annotator_groups = get_annotator_groups(annotations)

    #annotations = normalize_annotations(annotations)

    annotators = annotations.index.levels[0]

    corrs = defaultdict(lambda: defaultdict(list))

    def combs_generator(all_annotator_groups, limit):
        for _ in range(limit):
            random.shuffle(all_annotator_groups)
            yield all_annotator_groups[:len(all_annotator_groups) // 2]

    combs = combs_generator(all_annotator_groups, limit)

    for annotator_halfs in combs:
        annotator_half = tuple(it.chain(*annotator_halfs))
        select_mask = annotations.index.to_frame()["annotator"].isin(annotator_half)
        for score_name in score_names:
            p_corr, sq_err = compute_correlations_from_selection_mask(annotations, select_mask, score_name)
            corrs["pearson"][score_name].append(p_corr)
            corrs["sq_err"][score_name].append(sq_err)

    return corrs


def compute_annotator_srh(*args, **kwargs):
    corrs = compute_annotator_srh_raw(*args, **kwargs)
    results = defaultdict(dict)

    for corr_name, all_scores in corrs.items():
        for score_name, scores in all_scores.items():
            scores = np.array(scores)
            results[corr_name][score_name] = np.mean(scores)
            results[corr_name + "_var"][score_name] = np.var(scores)

    return results
