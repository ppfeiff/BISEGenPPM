import pandas as pd
import pm4py
from defaults import *


def load_bpic12():
    log = pm4py.read_xes("data/BPIC_2012/financial_log.xes")
    return log


def load_bpic13():
    log = pm4py.read_xes("data/BPIC_2013/bpi_challenge_incidents.xes")
    return log


def load_bpic17():
    log = pm4py.read_xes("data/BPIC_2017/bpic_2017.xes")
    return log

def load_helpdesk():
    log = pd.read_csv("data/Helpdesk/helpdesk.csv", sep=",")
    log = log.rename(columns={"CaseID": CASE_ID, "ActivityID": EVENT_ID, "CompleteTimestamp": TIMESTAMP})
    return log


def load_mobis():
    log = pd.read_csv("data/MobisChallenge2019/mobis_challenge_log_2019.csv", sep=";")
    log = log.rename(columns={"case": CASE_ID, "activity": EVENT_ID, "start": TIMESTAMP})
    return log



if __name__ == '__main__':
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 100000)
    pd.set_option('display.max_rows', 500)

    print(load_helpdesk())