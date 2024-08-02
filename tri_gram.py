from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline

import numpy as np
import pandas as pd

from defaults import CASE_ID, EVENT_ID, SPLIT_COLUMN

class TriGram():
    def __init__(self, log, seed):
        self.log = log
        self.seed = seed


    def prepare(self):

        self.log[EVENT_ID] = self.log[EVENT_ID].astype(str)
        activities = self.log[EVENT_ID].unique().tolist()
        self.act_to_int = {}
        for attr in activities:
            if attr not in self.act_to_int:
                idx = len(self.act_to_int)
                self.act_to_int[str(attr)] = str(idx)


        np.random.seed(self.seed)
        case_ids = self.log[CASE_ID].unique().tolist()

        train_ids = np.random.choice(case_ids, size=int(len(case_ids) * (1 - 0.2)), replace=False)

        train_idxs = self.log[self.log[CASE_ID].isin(train_ids)].index.tolist()
        test_idxs = self.log[~self.log[CASE_ID].isin(train_ids)].index.tolist()

        self.log[SPLIT_COLUMN] = pd.NA
        self.log.loc[train_idxs, SPLIT_COLUMN] = "TRAIN"
        self.log.loc[test_idxs, SPLIT_COLUMN] = "TEST"

        self.train_log = self.log[self.log[SPLIT_COLUMN] == "TRAIN"]
        self.test_log = self.log[self.log[SPLIT_COLUMN] == "TEST"]

        def build_samples(log):
            trace_groupby = log.groupby(CASE_ID)

            prefixes = []
            for c_id, trace in trace_groupby:
                cf = trace[EVENT_ID].tolist()
                for i in range(3, len(trace) + 1): #at least length 2 as prefix
                    sample = cf[0:i]
                    prefixes.append(sample)

            return prefixes

        self.train_log = build_samples(self.train_log)
        self.test_log = build_samples(self.test_log)


        def encode_prefixes(prefixes, act_to_int):
            # act_to_idx = build_activity_encoding(log)

            enc_prefixes = []
            for prefix in prefixes:
                enc_prefixes.append([act_to_int[x] if x in act_to_int else x for x in prefix])

            return enc_prefixes


        self.train_log = encode_prefixes(self.train_log, self.act_to_int)
        self.test_log = encode_prefixes(self.test_log, self.act_to_int)


    def train(self):
        train, vocab = padded_everygram_pipeline(3, self.train_log)
        self.model = KneserNeyInterpolated(order=3)
        self.model.fit(train, vocab)
        print(f"Vocab: {sorted(self.model.vocab)}")


    def test(self):
        correct_predictions = 0
        for sample in self.test_log:
            prefix = sample[-3:-1]
            target = sample[-1]

            #log_score = self.model.logscore(word=target[0], context=prefix)  # log base 2

            next_step = (None, -100)
            for token in self.model.vocab:
                score = self.model.score(word=token, context=prefix)
                if score > next_step[1]:
                    next_step = (token, score)

            if next_step[0] == target[0]:
                correct_predictions += 1

        accuracy = correct_predictions / len(self.test_log)
        print(accuracy)

        return accuracy

