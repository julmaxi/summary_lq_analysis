import numpy as np
import pandas as pd
import json


class OrdinalModel:
    def __init__(self, systems, coefficients, thresholds, annotator_covariance_matrix, document_covariance_matrix=None):
        self.systems = list(systems)
        self.coefficients = np.array(coefficients)
        self.thresholds = np.array(thresholds)
        self.annotator_covariance_matrix = np.array(annotator_covariance_matrix)
        self.document_covariance_matrix = np.array(document_covariance_matrix)

    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            data = json.load(f)

        annotator_matrix = data["random_effects"]["annotator"]

        annotator_matrix = np.array(annotator_matrix).reshape(len(data["system_names"]), len(data["system_names"]))
        document_matrix = None
        if "document" in data["random_effects"]:
            document_matrix = data["random_effects"]["document"]
            document_matrix = np.array(document_matrix).reshape(len(data["system_names"]), len(data["system_names"]))


        return cls(
            data["system_names"],
            np.array(data["coefficients"]),
            np.array(data["thresholds"]),
            annotator_matrix,
            document_matrix
        )
    def copy(self):
        return OrdinalModel(self.systems, self.coefficients, self.thresholds, self.annotator_covariance_matrix, self.document_covariance_matrix)

    def zero_coefficients(self):
        self.coefficients[:] = 0.

    def sample(self, design):
        annotators, documents = design
        annotator_errors = np.random.multivariate_normal(np.zeros(len(self.systems)), self.annotator_covariance_matrix, size=np.max(annotators) + 1)
        if self.document_covariance_matrix is not None:
            document_errors = np.random.multivariate_normal(np.zeros(len(self.systems)), self.document_covariance_matrix, size=np.max(documents) + 1)

        samples = []

        for idx, system in enumerate(self.systems):
            sampling_logits = self.thresholds - self.coefficients[idx]
            sampling_logits = sampling_logits.reshape(1, -1) - annotator_errors[annotators,0].reshape(-1, 1)
            if idx >= 1:
                sampling_logits -= annotator_errors[annotators,idx].reshape(-1, 1)

            if self.document_covariance_matrix is not None:
                if idx >= 1:
                    sampling_logits -= document_errors[documents,idx].reshape(-1, 1)
                sampling_logits -= document_errors[documents,0].reshape(-1, 1)


            sampling_odds = np.exp(sampling_logits)
            sampling_probabilities = sampling_odds / (sampling_odds + 1)
            rands = np.random.uniform(size=(len(annotators), 1))
            samples.append((rands > sampling_probabilities).sum(axis=1) + 1)

        samples_dfs = [pd.DataFrame.from_dict({"score": d}) for d in samples]
        for df in samples_dfs:
            df.index = pd.MultiIndex.from_arrays([annotators, documents], names=["annotator", "document"])
        join_df = pd.concat(samples_dfs, keys=self.systems, names=["system"])
        return join_df


def create_design(block_count, block_size, block_annotator_count):
    annotators = []
    documents = []

    curr_annotator = 0
    curr_document = 0
    for block_idx in range(block_count):
        block_doc_set = list(range(curr_document, curr_document + block_size))
        for block_annotator_idx in range(block_annotator_count):
            documents.extend(block_doc_set)
            annotators.extend([curr_annotator + block_annotator_idx] * len(block_doc_set))
        curr_annotator += block_annotator_count
        curr_document += block_size

    return np.array(annotators), np.array(documents)


thresholds_mn_likertd = [-5.2598, -4.3442, -3.0329, -1.7464, -0.5523, 1.0697]
annotator_covariance_mn_likertd = np.array([
    [1.0589, -0.397, 0.313, -0.570],
    [-0.397, 2.6498, -0.496, 0.876],
    [0.313, -0.496, 0.5877, -0.107],
    [-0.570, 0.876, -0.107, 4.0892]
])
covariates_mn_likertd = np.array([0.0, -3.4401, -0.2851, -3.7812])

covariates_mn_likertd_modified = np.array([0.0, -3.4401 + (0.352/2), -0.69, -3.7812 - (0.352/2)])


thresholds_cnndm_likertd = np.array([-4.1557, -2.7845, -1.5968, -0.3418, 1.0436, 2.6899])
document_covariance_cnndm_likertd = np.array([
    [+1.1083, -0.738, -0.680, -0.615, -0.816],
    [-0.738, +1.0900, +0.470, +0.512, +0.941],
    [-0.680, 0.470, 1.4792, 0.642, 0.722],
    [-0.615, 0.512, 0.642, 1.475, 0.710],
    [-0.816, 0.941, 0.722, 0.710, 1.9065]
])
annotator_covariance_cnndm_likertd = np.array([
    [1.7905, -0.503, -0.723, -0.567, -0.536],
    [-0.503, 0.3533, 0.578, 0.517, 0.339],
    [-0.723,  0.578, 1.7695, 0.833,  0.736],
    [-0.567,  0.517,  0.833, 0.9142, 0.240],
    [-0.536, 0.339, 0.736, 0.240, 0.9160]
])
covariates_cnndm_likertd = [0.0, -0.7096, 1.3953, 0.6771, -1.2513]


annotator_covariance_mn_ranking = np.array([
    [0.2163, -0.746, -0.686, -0.084],
    [-0.746, 1.8462,  0.105, -0.431],
    [-0.686, 0.105, 0.9569, 0.247],
    [-0.084, -0.431, 0.247, 0.3873]
])

threshold_mn_rank = np.array([-2.8417, -1.2238, 0.2938])
covariates_mn_rank = np.array([0, -2.0126, -0.2246, -2.9426])
covariates_mn_rank_modified = np.array([0, -2.0126 - 0.12, -0.69, -2.9426 + 0.12])


MODELS = {
    "likertD:multi_news": OrdinalModel(
        ["__REFERENCE__", "abssentrw", "BART", "onmt_pg"],
        covariates_mn_likertd,
        thresholds_mn_likertd,
        annotator_covariance_mn_likertd
    ),
    "ranking:multi_news": OrdinalModel(
        ["__REFERENCE__", "abssentrw", "BART", "onmt_pg"],
        covariates_mn_rank,
        threshold_mn_rank,
        annotator_covariance_mn_ranking
    ),
    "ranking:multi_news:modified": OrdinalModel(
        ["__REFERENCE__", "abssentrw", "BART", "onmt_pg"],
        covariates_mn_rank_modified,
        threshold_mn_rank,
        annotator_covariance_mn_ranking
    ),
    "likertD:multi_news:modified": OrdinalModel(
        ["__REFERENCE__", "abssentrw", "BART", "onmt_pg"],
        covariates_mn_likertd_modified,
        thresholds_mn_likertd,
        annotator_covariance_mn_likertd
    ),
    "likertD:cnn_dailymail": OrdinalModel(
        ["__REFERENCE__", "abssentrw", "BART", "onmt_pg", "seneca"],
        covariates_cnndm_likertd,
        thresholds_cnndm_likertd,
        annotator_covariance_cnndm_likertd,
        document_covariance_cnndm_likertd
    )
}

if __name__ == "__main__":
    sample = MODELS["ranking:multi_news:modified"].sample(create_design(50, 5, 1))
    sample.to_csv("sample.csv")
