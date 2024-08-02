import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from itertools import permutations, product

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 12)
pd.options.display.float_format = "{:.2f}".format

# Random generator
seed = 9999

rng = np.random.default_rng(seed=seed)
random.seed(seed)

def sample_timestamp(start, end):
    delta = int((end - start).total_seconds())
    random_seconds = float(rng.integers(0, delta, endpoint=True))
    random_time = start + timedelta(seconds=random_seconds)
    return random_time

# add a number of minutes to a timestamp
def add_time(a: datetime, b: int):
    return a + timedelta(minutes=b)

prices = rng.integers(10, 1000, 100)
prices2 = rng.integers(10, 10000, 1000)
resources = [f"R{i}" for i in range(1, 101)]
more_resources = [f"R{i}" for i in range(1, 1001)]
start_time = datetime.strptime("2015-01-01 09:00", "%Y-%m-%d %H:%M")
end_time = datetime.strptime("2019-12-01 16:00", "%Y-%m-%d %H:%M")

n_cases = 10000


# Scenario CF1: parallel, prediction after parallel block

# simple: 5x1
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])
perm = np.array(list(permutations(["D", "E", "F", "G", "H"])))
rng.shuffle(perm)

split_idx = int(0.8 * len(perm))
train = perm[:split_idx]
test = perm[split_idx:]

n_events = 11

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]

    perm_idx = rng.integers(0, len(perm))
    activities.extend(perm[perm_idx])

    if perm_idx >= split_idx:
        case_label = "test"
    else:
        case_label = "train"

    activities.extend(["I", "J", "K"])

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)
    price = rng.choice(prices, n_events)

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": [case_label] * n_events})

    df = pd.concat([df, case_df], axis=0)

df["target"] = df["activity"].isin(["I", "J", "K"])

df.to_csv(r"scenario_logs/log_CF1_parallel_simple.csv", index=False)


# advanced: 3x3
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])

def generate_interleavings(prefix, remaining):
    if not any(remaining):
        return [prefix]

    interleavings = []
    for i in range(len(remaining)):
        if remaining[i]:
            # Create a new prefix by adding the first element of the current branch
            new_prefix = prefix + [remaining[i][0]]
            # Create a new remaining list where the first element of the current branch is removed
            new_remaining = remaining[:i] + [remaining[i][1:]] + remaining[i + 1:]
            # Recurse with the new prefix and remaining
            interleavings.extend(generate_interleavings(new_prefix, new_remaining))
    return interleavings

def all_valid_interleavings(branches):
    return generate_interleavings([], branches)

branch1 = ["D", "E", "F"]
branch2 = ["G", "H", "I"]
branch3 = ["J", "K", "L"]

perm_advanced = all_valid_interleavings([branch1, branch2, branch3])
rng.shuffle(perm_advanced)

split_idx = int(0.8 * len(perm_advanced))
train = perm_advanced[:split_idx]
test = perm_advanced[split_idx:]

n_events = 15

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]

    perm_idx = rng.integers(0, len(perm_advanced))
    activities.extend(perm_advanced[perm_idx])

    if perm_idx >= split_idx:
        case_label = "test"
    else:
        case_label = "train"

    activities.extend(["M", "N", "O"])

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)
    price = rng.choice(prices, n_events)

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": case_label})

    df = pd.concat([df, case_df], axis=0)
    df = df.reset_index(drop=True)


df["target"] = df["activity"].isin(["M", "N", "O"])

df.to_csv(r"scenario_logs/log_CF1_parallel_advanced.csv", index=False)


# Scenario CF2: parallel, prediction in parallel block

# simple: 5x1
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])

branch1 = ["D"]
branch2 = ["E"]
branch3 = ["F"]
branch4 = ["G"]
branch5 = ["H"]

perm = all_valid_interleavings([branch1, branch2, branch3, branch4, branch5])

def generate_prefixes(sequence):
    return [sequence[:i+1] for i in range(len(sequence))]

prefs = [prefix for sequence in perm for prefix in generate_prefixes(sequence)]

tuples = [tuple(lst) for lst in prefs]
unique_tuples = list(set(tuples))
unique_prefs = [list(tpl) for tpl in unique_tuples]

# noinspection PyTypeChecker
rng.shuffle(unique_prefs)

split_idx = int(0.8 * len(unique_prefs))
train = unique_prefs[:split_idx]
test = unique_prefs[split_idx:]
train_set = {tuple(t) for t in train}
test_set = {tuple(t) for t in test}

n_events = 11

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]

    perm_idx = rng.integers(0, len(perm))
    activities.extend(perm[perm_idx])

    activities.extend(["I", "J", "K"])

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)
    price = rng.choice(prices, n_events)

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": None})

    df = pd.concat([df, case_df], axis=0)
    df = df.reset_index(drop=True)


train_df = pd.DataFrame()
test_df = pd.DataFrame()

# Unique identifier counters
prefix_id_counter = 1
unique_id_counter = 1

# Dictionary to store unique sequences and their assigned unique_id
unique_sequences = {}

for case_name, case in df.groupby('case'):
    print(case_name)
    par = case.iloc[3:8]
    for j in range(len(par)):
        current_prefix = tuple(par.activity[:j])

        if current_prefix not in unique_sequences:
            unique_sequences[current_prefix] = unique_id_counter
            unique_id_counter += 1

        unique_id = unique_sequences[current_prefix]
        prefix_id = prefix_id_counter
        prefix_id_counter += 1

        if current_prefix in train_set:
            # df.loc[par.index[j], 'target_train'] = True
            prefix_df = case.iloc[:j + 4].copy()
            prefix_df['prefix_id'] = prefix_id
            prefix_df['unique_id'] = unique_id
            train_df = pd.concat([train_df, prefix_df])
        elif current_prefix in test_set:
            # df.loc[par.index[j], 'target_test'] = True
            prefix_df = case.iloc[:j + 4].copy()
            prefix_df['prefix_id'] = prefix_id
            prefix_df['unique_id'] = unique_id
            test_df = pd.concat([test_df, prefix_df])

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

train_df.to_csv(r"scenario_logs/TRAIN_log_CF2_parallel_simple.csv", index=False)
test_df.to_csv(r"scenario_logs/TEST_log_CF2_parallel_simple.csv", index=False)

# df.loc[df.groupby('case').cumcount() == 3, 'target_CF2'] = False  # first parallel activity is not a target
df.to_csv(r"scenario_logs/log_CF2_parallel_simple.csv", index=False)


# advanced: 3x3
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])

branch1 = ["D", "E", "F"]
branch2 = ["G", "H", "I"]
branch3 = ["J", "K", "L"]

perm_advanced = all_valid_interleavings([branch1, branch2, branch3])

def generate_prefixes(sequence):
    return [sequence[:i+1] for i in range(len(sequence))]

prefs = [prefix for sequence in perm_advanced for prefix in generate_prefixes(sequence)]

tuples = [tuple(lst) for lst in prefs]
unique_tuples = list(set(tuples))
unique_prefs = [list(tpl) for tpl in unique_tuples]

# noinspection PyTypeChecker
rng.shuffle(unique_prefs)

split_idx = int(0.8 * len(unique_prefs))
train = unique_prefs[:split_idx]
test = unique_prefs[split_idx:]
train_set = {tuple(t) for t in train}
test_set = {tuple(t) for t in test}

n_events = 15

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]

    perm_idx = rng.integers(0, len(perm_advanced))
    activities.extend(perm_advanced[perm_idx])

    activities.extend(["M", "N", "O"])

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)
    price = rng.choice(prices, n_events)

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": None})

    df = pd.concat([df, case_df], axis=0)
    df = df.reset_index(drop=True)

train_df = pd.DataFrame()
test_df = pd.DataFrame()

# Unique identifier counters
prefix_id_counter = 1
unique_id_counter = 1

# Dictionary to store unique sequences and their assigned unique_id
unique_sequences = {}


for case_name, case in df.groupby('case'):
    print(case_name)
    par = case.iloc[3:12]
    for j in range(len(par)):
        current_prefix = tuple(par.activity[:j])

        if current_prefix not in unique_sequences:
            unique_sequences[current_prefix] = unique_id_counter
            unique_id_counter += 1

        unique_id = unique_sequences[current_prefix]
        prefix_id = prefix_id_counter
        prefix_id_counter += 1

        if current_prefix in train_set:
            # df.loc[par.index[j], 'target_train'] = True
            prefix_df = case.iloc[:j + 4].copy()
            prefix_df['prefix_id'] = prefix_id
            prefix_df['unique_id'] = unique_id
            train_df = pd.concat([train_df, prefix_df])
        elif current_prefix in test_set:
            # df.loc[par.index[j], 'target_test'] = True
            prefix_df = case.iloc[:j + 4].copy()
            prefix_df['prefix_id'] = prefix_id
            prefix_df['unique_id'] = unique_id
            test_df = pd.concat([test_df, prefix_df])

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

train_df.to_csv(r"scenario_logs/TRAIN_log_CF2_parallel_advanced.csv", index=False)
test_df.to_csv(r"scenario_logs/TEST_log_CF2_parallel_advanced.csv", index=False)


df.to_csv(r"scenario_logs/log_CF2_parallel_advanced.csv", index=False)


# Scenario CF3: loop, prediction after loop

# simple: 1 activity, loops up to 5 times
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])
max_loops = 25
loop_counts = np.array(list(range(1, max_loops + 1)))
rng.shuffle(loop_counts)

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]

    loop_count_idx = rng.integers(0, len(loop_counts))
    loop_count = loop_counts[loop_count_idx]
    for i in range(loop_count):
        activities.extend(["D"])

    if loop_count_idx < 0.2 * max_loops:
        case_label = "test"
    else:
        case_label = "train"

    activities.extend(["E", "F", "G"])
    n_events = len(activities)

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)
    price = rng.choice(prices, n_events)

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": [case_label] * n_events})

    df = pd.concat([df, case_df], axis=0)

df["target"] = df["activity"].isin(["F", "G"])
df.to_csv(r"scenario_logs/log_CF3_loop_simple.csv", index=False)


# advanced: 2x2 activities, each can loop up to 5 times
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])
max_loops = 10
loop_counts = np.array(list(range(1, max_loops + 1)))
loop_perm = list(product(loop_counts, loop_counts))
rng.shuffle(loop_perm)

for case in range(1, n_cases + 1):

    activities = ["A", "B"]

    loop_count_idx = rng.integers(0, len(loop_perm))
    loop_count = loop_perm[loop_count_idx]
    for i in range(loop_count[0]):
        activities.extend(["C", "D"])

    activities.extend("E")

    for i in range(loop_count[1]):
        activities.extend(["F", "G"])

    if loop_count_idx < len(loop_perm) // 5:
        case_label = "test"
    else:
        case_label = "train"

    activities.extend(["H", "I"])
    n_events = len(activities)

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)
    price = rng.choice(prices, n_events)

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": [case_label] * n_events})

    df = pd.concat([df, case_df], axis=0)

df["target"] = df["activity"].isin(["I"])
df.to_csv(r"scenario_logs/log_CF3_loop_advanced.csv", index=False)


# Scenario CF4: loop, prediction in loop

# simple: 1 activity, loops up to 5 times
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])
max_loops = 25
loop_counts = np.array(list(range(1, max_loops + 1)))

loops = [["D"] * (i + 1) for i in range(max_loops)]
rng.shuffle(loops)

split_idx = int(0.8 * len(loops))
train = loops[:split_idx]
test = loops[split_idx:]
train_set = {tuple(t) for t in train}
test_set = {tuple(t) for t in test}

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]

    loop_count_idx = rng.integers(0, len(loop_counts))
    loop_count = loop_counts[loop_count_idx]
    for i in range(loop_count):
        activities.extend(["D"])

    activities.extend(["E", "F", "G"])
    n_events = len(activities)

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)
    price = rng.choice(prices, n_events)

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": None})

    df = pd.concat([df, case_df], axis=0)
    df = df.reset_index(drop=True)

df["target_train"] = False
df["target_test"] = False

for case_name, case in df.groupby('case'):
    start_index = case.index[3]
    end_index = case[case['activity'] == 'E'].index[0]
    par = case.loc[start_index:end_index]      # this is inclusive, hence E as the end activity
    for j in range(len(par)):
        current_prefix = tuple(par.activity[:j])
        if current_prefix in train_set:
            df.loc[par.index[j], 'target_train'] = True
        elif current_prefix in test_set:
            df.loc[par.index[j], 'target_test'] = True

df.to_csv(r"scenario_logs/log_CF4_loop_simple.csv", index=False)


# advanced: 2x2 activities, each can loop up to 5 times
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])
max_loops = 10
loop_counts = np.array(list(range(1, max_loops + 1)))
loop_perm = list(product(loop_counts, loop_counts))
loops_advanced = [["C", "D"] * i[0] + ["E"] + ["F", "G"] * i[1] for i in loop_perm]

prefs = [prefix for sequence in loops_advanced for prefix in generate_prefixes(sequence)]
tuples = [tuple(lst) for lst in prefs]
unique_tuples = list(set(tuples))
unique_prefs = [list(tpl) for tpl in unique_tuples]

split_idx = int(0.8 * len(unique_prefs))
train = unique_prefs[:split_idx]
test = unique_prefs[split_idx:]
train_set = {tuple(t) for t in train}
test_set = {tuple(t) for t in test}

for case in range(1, n_cases + 1):

    activities = ["A", "B"]

    loop_count_idx = rng.integers(0, len(loop_perm))
    loop_count = loop_perm[loop_count_idx]
    for i in range(loop_count[0]):
        activities.extend(["C", "D"])

    activities.extend("E")

    for i in range(loop_count[1]):
        activities.extend(["F", "G"])

    activities.extend(["H", "I"])
    n_events = len(activities)

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)
    price = rng.choice(prices, n_events)

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": None})

    df = pd.concat([df, case_df], axis=0)

    df = df.reset_index(drop=True)


df["target_train"] = False
df["target_test"] = False

for case_name, case in df.groupby('case'):
    start_index = case.index[2]
    end_index = case[case['activity'] == 'H'].index[0]
    par = case.loc[start_index:end_index]  # this is inclusive, hence E as the end activity
    for j in range(len(par)):
        current_prefix = tuple(par.activity[:j])
        if current_prefix in train_set:
            df.loc[par.index[j], 'target_train'] = True
        elif current_prefix in test_set:
            df.loc[par.index[j], 'target_test'] = True

df.to_csv(r"scenario_logs/log_CF4_loop_advanced.csv", index=False)


# Scenario ATT1: activity/resource combination

# simple: new activity + resource combination in one event (C), 100 resources
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])
train_resources = rng.choice(resources, int(0.8 * len(resources)), replace=False)      # 80 / 100

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]
    price = rng.choice(prices, 7)

    if price[2] < 500:     # price of C
        activities.extend("D")
    else:
        activities.extend("E")

    activities.extend(["F", "G", "H"])
    n_events = len(activities)

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)

    if res[2] in train_resources:
        case_label = "train"
    else:
        case_label = "test"

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": [case_label] * n_events})

    df = pd.concat([df, case_df], axis=0)

df["target"] = df["activity"].isin(["D", "E", "F", "G", "H"])
df.to_csv(r"scenario_logs/log_ATT1_resource_combination_simple.csv", index=False)



# advanced: new activity + resource combination in one event (C), 1000 resources
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])
train_resources = rng.choice(more_resources, int(0.8 * len(more_resources)), replace=False)     # 800 / 1000

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]
    price = rng.choice(prices, 7)

    if price[2] < 500:     # price of C
        activities.extend("D")
    else:
        activities.extend("E")

    activities.extend(["F", "G", "H"])
    n_events = len(activities)

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(more_resources, n_events)

    if res[2] in train_resources:
        case_label = "train"
    else:
        case_label = "test"

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": [case_label] * n_events})

    df = pd.concat([df, case_df], axis=0)

df["target"] = df["activity"].isin(["D", "E", "F", "G", "H"])
df.to_csv(r"scenario_logs/log_ATT1_resource_combination_advanced.csv", index=False)


# Scenario ATT2: activity/price combination

# simple: new activity + price combination in one event (C), prices 10 - 1000, 100 possible values
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])
train_prices = rng.choice(prices, int(0.8 * len(prices)), replace=False)    # 80 / 100

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]
    price = rng.choice(prices, 7)

    if price[2] < 500:     # price of C
        activities.extend("D")
    else:
        activities.extend("E")

    activities.extend(["F", "G", "H"])
    n_events = len(activities)

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)

    if price[2] in train_prices:
        case_label = "train"
    else:
        case_label = "test"

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": [case_label] * n_events})

    df = pd.concat([df, case_df], axis=0)

df["target"] = df["activity"].isin(["D", "E"])
df.to_csv(r"scenario_logs/log_ATT2_price_combination_simple.csv", index=False)


# advanced: new combination of prices in one event (C), prices 10 - 50000, 100 possible values
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])
train_prices2 = rng.choice(prices2, int(0.8 * len(prices2)), replace=False)     # 800 / 1000

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]
    price = rng.choice(prices2, 7)

    if price[2] < 5000:     # price of C, threshold changed to 5000 so that about half are below and above
        activities.extend("D")
    else:
        activities.extend("E")

    activities.extend(["F", "G", "H"])
    n_events = len(activities)

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)

    if price[2] in train_prices2:
        case_label = "train"
    else:
        case_label = "test"

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": [case_label] * n_events})

    df = pd.concat([df, case_df], axis=0)

df["target"] = df["activity"].isin(["D", "E"])
df.to_csv(r"scenario_logs/log_ATT2_price_combination_advanced.csv", index=False)


# Scenario CF5: unseen activity

# simple: replace one activity (C)
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]
    price = rng.choice(prices, 7)

    if price[2] < 500:  # price of C
        activities.extend("D")
    else:
        activities.extend("E")

    activities.extend(["F", "G", "H"])

    n_events = len(activities)

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)

    if rng.random() < 0.2:
        activities[2] = "X"
        case_label = "test"
    else:
        case_label = "train"

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": [case_label] * n_events})

    df = pd.concat([df, case_df], axis=0)

df["target"] = df["activity"].isin(["D", "E", "F", "G", "H"])
df.to_csv(r"scenario_logs/log_CF5_unseen_activity_simple.csv", index=False)


# advanced: replace two activities (B and C)
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]
    price = rng.choice(prices, 7)

    if price[2] < 500:  # price of C
        activities.extend("D")
    else:
        activities.extend("E")

    activities.extend(["F", "G", "H"])

    n_events = len(activities)

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)

    if rng.random() < 0.2:
        activities[1] = "X"
        activities[2] = "Y"
        case_label = "test"
    else:
        case_label = "train"

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": [case_label] * n_events})

    df = pd.concat([df, case_df], axis=0)

df["target"] = df["activity"].isin(["D", "E", "F", "G", "H"])
df.to_csv(r"scenario_logs/log_CF5_unseen_activity_advanced.csv", index=False)


# Scenario ATT3: unseen resource

# simple: replace 1 resource (C)
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]
    price = rng.choice(prices, 7)

    if price[2] < 500:  # price of C
        activities.extend("D")
    else:
        activities.extend("E")

    activities.extend(["F", "G", "H"])

    n_events = len(activities)

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)

    if rng.random() < 0.2:
        res[2] = "R999"
        case_label = "test"
    else:
        case_label = "train"

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": [case_label] * n_events})

    df = pd.concat([df, case_df], axis=0)

df["target"] = df["activity"].isin(["D", "E", "F", "G", "H"])
df.to_csv(r"scenario_logs/log_ATT3_unseen_resource_simple.csv", index=False)


# advanced: replace 2 resources (B and C)
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]
    price = rng.choice(prices, 7)

    if price[2] < 500:  # price of C
        activities.extend("D")
    else:
        activities.extend("E")

    activities.extend(["F", "G", "H"])

    n_events = len(activities)

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)

    if rng.random() < 0.2:
        res[1] = "R999"
        res[2] = "R998"
        case_label = "test"
    else:
        case_label = "train"

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": [case_label] * n_events})

    df = pd.concat([df, case_df], axis=0)

df["target"] = df["activity"].isin(["D", "E", "F", "G", "H"])
df.to_csv(r"scenario_logs/log_ATT3_unseen_resource_advanced.csv", index=False)


# Scenario ATT4: unseen price

# simple: replace one price (C)
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]
    price = rng.choice(prices, 7)

    if price[2] < 500:  # price of C
        activities.extend("D")
    else:
        activities.extend("E")

    activities.extend(["F", "G", "H"])

    n_events = len(activities)

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)

    if rng.random() < 0.2:
        price[2] += 50
        case_label = "test"
    else:
        case_label = "train"

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": [case_label] * n_events})

    df = pd.concat([df, case_df], axis=0)

df["target"] = df["activity"].isin(["D", "E"])
df.to_csv(r"scenario_logs/log_ATT4_unseen_price_simple.csv", index=False)


# advanced: replace two prices (B and C)
df = pd.DataFrame(columns=["case", "activity", "timestamp", "resource", "price", "case_label"])

for case in range(1, n_cases + 1):

    activities = ["A", "B", "C"]
    price = rng.choice(prices, 7)

    if price[2] < 500:  # price of C
        activities.extend("D")
    else:
        activities.extend("E")

    activities.extend(["F", "G", "H"])

    n_events = len(activities)

    case_start_time = sample_timestamp(start_time, end_time)
    intervals = rng.uniform(0, 48 * 3600, n_events - 1)
    time_deltas = pd.to_timedelta(intervals, unit='s')
    timestamps = [case_start_time]
    for delta in time_deltas:
        timestamps.append(timestamps[-1] + delta)

    res = rng.choice(resources, n_events)

    if rng.random() < 0.2:
        price[1] += 50
        price[2] += 50
        case_label = "test"
    else:
        case_label = "train"

    case_df = pd.DataFrame({"case": [case] * n_events, "activity": activities, "timestamp": timestamps, "resource": res,
                            "price": price, "case_label": [case_label] * n_events})

    df = pd.concat([df, case_df], axis=0)

df["target"] = df["activity"].isin(["D", "E"])
df.to_csv(r"scenario_logs/log_ATT4_unseen_price_advanced.csv", index=False)

