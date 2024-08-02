import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle


from defaults import *
from load_data import load_bpic12, load_helpdesk, load_mobis, load_bpic17, load_bpic13
from tri_gram import TriGram



SEEDS = [1111, 2222, 3333, 4444, 5555, None]
MPPN_values = [0.9, 0.9, 0.9, 0.9, 0.9]




def split_event_log_by_seed(log, seed, ratio):
    if seed == None:
        return split_event_log_by_time(log, ratio)

    np.random.seed(seed)
    case_ids = log[CASE_ID].unique().tolist()

    train_ids = np.random.choice(case_ids, size=int(len(case_ids) * (1 - ratio)), replace=False)

    train_idxs = log[log[CASE_ID].isin(train_ids)].index.tolist()
    test_idxs = log[~log[CASE_ID].isin(train_ids)].index.tolist()

    log[SPLIT_COLUMN] = pd.NA
    log.loc[train_idxs, SPLIT_COLUMN] = "TRAIN"
    log.loc[test_idxs, SPLIT_COLUMN] = "TEST"

    #print(log)

    return log


def split_event_log_by_time(log, ratio):
    cases = log[CASE_ID].unique().tolist()

    train_case_ids = cases[0:int(len(cases) * (1 - ratio))]
    test_case_ids = cases[int(len(cases) * (1 - ratio)):]

    train_idxs = log[log[CASE_ID].isin(train_case_ids)].index.tolist()
    test_idxs = log[~log[CASE_ID].isin(train_case_ids)].index.tolist()

    log[SPLIT_COLUMN] = pd.NA
    log.loc[train_idxs, SPLIT_COLUMN] = "TRAIN"
    log.loc[test_idxs, SPLIT_COLUMN] = "TEST"

    return log


def build_activity_encoding(log):
    activities = log[EVENT_ID].unique().tolist()

    act_to_int = {}
    for attr in activities:
        if attr not in act_to_int:
            idx = len(act_to_int)
            act_to_int[str(attr)] = idx

    return act_to_int


def build_samples(log):
    trace_groupby = log.groupby(CASE_ID)

    prefixes = []
    for c_id, trace in trace_groupby:
        cf = trace[EVENT_ID].tolist()
        for i in range(2, len(trace) + 1):
            prefixes.append(cf[0:i])

    #print(len(prefixes))
    return prefixes


def encode_prefixes(prefixes, act_to_int):
    #act_to_idx = build_activity_encoding(log)

    enc_prefixes = []
    for prefix in prefixes:
        enc_prefixes.append([act_to_int[x] for x in prefix])

    return enc_prefixes


def find_unique_prefixes(prefixes):
    unique_prefixes = []
    for prefix in prefixes:
        if prefix not in unique_prefixes:
            unique_prefixes.append(prefix)

    #print(len(unique_prefixes))
    return unique_prefixes




def calculate_prefix_leakage_cf(el_dset, seed=None, split_ratio=0.2):

    el_dset[EVENT_ID] = el_dset[EVENT_ID].astype(str)
    el_dset = split_event_log_by_seed(el_dset, seed=seed, ratio=split_ratio)
    cat_to_int = build_activity_encoding(el_dset)

    train_split = el_dset[el_dset[SPLIT_COLUMN] == "TRAIN"]
    test_split = el_dset[el_dset[SPLIT_COLUMN] == "TEST"]

    train_samples = build_samples(train_split)
    test_samples = build_samples(test_split)

    train_samples = encode_prefixes(train_samples, cat_to_int)
    train_prefixes = [sample[0:-1] for sample in train_samples]
    unique_train_prefixes = find_unique_prefixes(train_prefixes)

    test_samples = encode_prefixes(test_samples, cat_to_int)
    test_prefixes = [sample[0:-1] for sample in test_samples]

    num_leaked_prefixes = 0
    leaked_prefixes = []
    for test_prefix in test_prefixes:
        if test_prefix in unique_train_prefixes:
            num_leaked_prefixes += 1
            if test_prefix not in leaked_prefixes:
                leaked_prefixes.append(test_prefix)

    leaked_percentage = num_leaked_prefixes / len(test_samples)

    return leaked_percentage, leaked_prefixes, test_prefixes


def build_test_cf_samples(el_dset, seed):
    el_dset[EVENT_ID] = el_dset[EVENT_ID].astype(str)
    el_dset = split_event_log_by_seed(el_dset, seed=seed, ratio=0.2)
    #cat_to_int = build_activity_encoding(el_dset)

    test_split = el_dset[el_dset[SPLIT_COLUMN] == "TEST"]
    test_samples = build_samples(test_split)

    return test_samples


def prefix_label_distribution(test_samples):
    pairs = []
    for sample in test_samples:
        prefix_enc = tuple(sample[0:-1])
        label = sample[-1]
        pairs.append([prefix_enc, label])

    pairs = pd.DataFrame(pairs)
    pairs = pairs.rename({0: "prefix", 1: "label"}, axis=1)

    pld_ = {}
    for pref_id, pref in pairs.groupby("prefix"):
        pld_[pref_id] = pref["label"].value_counts(normalize=False)

    return pld_


def calculate_max_accuracy(el_dset, seed):
    test_samples = build_test_cf_samples(el_dset, seed)
    test_pld = prefix_label_distribution(test_samples)

    max_correct_predictions = 0
    for _, labels_ in test_pld.items():
        max_correct_predictions += labels_.values[0]
        # assumption: for each prefix, the most common label is always predicted

    max_acc = max_correct_predictions / len(test_samples)
    return max_acc


def train_trigram(el_dset, seed):
    ngram = TriGram(el_dset, seed)
    ngram.prepare()
    ngram.train()
    accuracy = ngram.test()
    return accuracy


def train_mppn(el_dset, seed):
    mppn = MPPNCategoricalNSP(el_dataset=el_dset,
                              **MPPN_KWARGS[el_dset.name],
                              target_attribute=XESTerminology.EVENT_ID.value,
                              representation_model_path=None,
                              modelname=f"MPPN_{el_dset.name}_{seed}_baseline")

    callbacks = [
        EarlyStopping(
            monitor="eval_loss",
            mode='min',
            patience=5,
            min_delta=1e-2,
            verbose=1,
        ),
        SaveModelCheckpoint(
            filename=mppn.modelname,
            monitor="eval_loss",
            mode="min"),
    ]

    mppn.prepare(seed=seed,
                 representation_dim=128,
                 mlp_num_layers=2,
                 input_length=64,
                 concat=True,
                 persistent_workers=True,
                 min_prefix_length=2,
                 callbacks=callbacks)

    mppn.train(epochs=100)
    mppn.load_checkpoint()
    mppn.save_model()
    mppn.load_model()
    test_results = mppn.test()
    return test_results["test_accuracy"]






def compute_metrics_for_all_logs():
    el_helpdesk = load_helpdesk()
    el_bpic12 = load_bpic12()
    el_bpic13 = load_bpic13()
    el_bpic17 = load_bpic17()
    el_mobis = load_mobis()

    #dsets = [("bpic12", el_bpic12), ("bpic13", el_bpic13), ("bpic17", el_bpic17), ("helpdesk", el_helpdesk), ("mobis", el_mobis)]
    dsets = [("bpic12", el_bpic12), ("bpic13", el_bpic13), ("helpdesk", el_helpdesk), ("mobis", el_mobis)]

    leakages = {}
    max_accuracy = {}
    ngram_baseline = {}
    mppn = {}

    for el_dset in dsets:
        if el_dset[0] not in leakages.keys():
            leakages[el_dset[0]] = []
            max_accuracy[el_dset[0]] = []
            ngram_baseline[el_dset[0]] = []
            mppn[el_dset[0]] = []

        for seed in SEEDS:
            leakages[el_dset[0]].append(calculate_prefix_leakage_cf(el_dset[1], seed=seed)[0])
            max_accuracy[el_dset[0]].append(calculate_max_accuracy(el_dset[1], seed=seed))
            ngram_baseline[el_dset[0]].append(train_trigram(el_dset[1], seed=seed))
            #mppn[el_dset.name].append(train_mppn(el_dset.__copy__(), seed=seed))

    print(leakages)
    print(max_accuracy)
    print(ngram_baseline)
    print(mppn)

    with open("leakages.pkl", "wb") as fp:
        pickle.dump(leakages, fp)

    with open("max_acc.pkl", "wb") as fp:
        pickle.dump(max_accuracy, fp)

    with open("ngram_baseline.pkl", "wb") as fp:
        pickle.dump(ngram_baseline, fp)

    with open("mppn.pkl", "wb") as fp:
        pickle.dump(mppn, fp)

    pd_results = pd.DataFrame(data=[leakages, max_accuracy, ngram_baseline, mppn])
    print(pd_results)
    #pd_results.to_csv("all_results.csv", index=False, sep=";")

    return leakages, max_accuracy, ngram_baseline, mppn





def plot(metrics):
    leakages = metrics[0]
    max_accuracies = metrics[1]
    ngram_baselines = metrics[2]
    mppn = metrics[3]

    fig, axs = plt.subplots(1, len(leakages.keys()), figsize=(12, 5), dpi=800, sharey="all")
    axs[0].set_ylim(0, 1)
    xval = np.linspace(0, 1, len(leakages["Helpdesk"]))

    for log_name, ax in zip(leakages.keys(), axs):
        ax.scatter(xval, leakages[log_name], label="Prefix Leakage", marker="o", c="blue")
        ax.scatter(xval, ngram_baselines[log_name], label="Ngram", marker='s', c="#6baed6")
        ax.scatter(xval, mppn[log_name], label="MPPN", marker='^', c="#1f77b4")
        ax.scatter(xval, max_accuracies[log_name], label="Accuracy \nlimit", marker='_', c="red", s=100)
        ax.set_title(log_name, y=-0.1, fontproperties=axs[0].yaxis.label.get_fontproperties())
        ax.set_xlim(-0.15, 1.15)
        ax.set_xticks([])
        if log_name == "BPIC12":
            ax.yaxis.set_ticks_position('left')
            ax.set_ylabel('Accuracy')
        else:
            ax.yaxis.set_ticks_position('none')
            ax.spines['left'].set_color('#CCCCCC')
        ax.set_xticklabels([])
        ax.grid()

    plt.legend(bbox_to_anchor=(0.1, 0.3), loc='upper left')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)

    plt.savefig("validity_issues_v2.png")
    plt.show()





if __name__ == '__main__':

    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 100000)
    pd.set_option('display.max_rows', 500)

    #dset_helpdesk = load_helpdesk()
    #train_trigram(dset_helpdesk, 1111)
    #exit(99)


    metrics = compute_metrics_for_all_logs()
    #metrics = load_metrics_from_platte()
    plot(metrics)
    exit(99)

    dset_bpic17 = load_bpic17()
    dset_helpdesk = load_helpdesk()
    dset_bpic12 = load_bpic12()
    dset_bpic13 = load_bpic13()
    dset_mobis = load_mobis()

    print(calculate_max_accuracy(el_dset=dset_helpdesk, seed=SEEDS[0]))
    print(calculate_max_accuracy(el_dset=dset_mobis, seed=SEEDS[0]))
    print(calculate_max_accuracy(el_dset=dset_bpic12, seed=SEEDS[0]))
    print(calculate_max_accuracy(el_dset=dset_bpic13, seed=SEEDS[0]))


    for seed in SEEDS:
        print(calculate_prefix_leakage_cf(el_dset=dset_helpdesk, seed=seed)[0])
        print(calculate_prefix_leakage_cf(el_dset=dset_bpic12, seed=seed)[0])
        print(calculate_prefix_leakage_cf(el_dset=dset_bpic13, seed=seed)[0])

    exit(99)

