"""
parse_data.py

Read all data in all folders. Shuffle, partition into test and train sets, and
save data into separate CSVs for later use.
Only reads data from 'normalized reads' column.

Hermon Mulat, Collin Epstein
2/26/18
CSC371
Dr. Ramanujan
"""

import sys, os
import pandas as pd
from sklearn.utils import shuffle

MANIFEST = "MANIFEST.txt"
BASE = "./data/"

CATEGORY_MAPPING = {}

def read_row(full_path):
    """
    Read in feature data from single data file
    """

    row = []
    with open(full_path,'r') as row_f:
        row_f.next()
        for line in row_f:
            row.append(line.split()[2])

    return row

def read_folder(fn):
    """
    Read in data from all files in a give folder
    """

    all_rows = []
    folder_mani = fn + MANIFEST
    with open(folder_mani,"r") as fm:
        fm.next()
        for line in fm:
            if 'annotation' not in line: # exclude annotation files
                filename = fn+line.split()[1]
                row = read_row(filename)
                all_rows.append(row)

    return all_rows

def parse_data():
    """
    Parse all data and save as training and testing sets
    """

    count = 0
    DATA = pd.DataFrame()

    for x in os.listdir(BASE):
        if (".DS_Store" not in x):
            fld_path = BASE+x+"/"
            CATEGORY_MAPPING[x] = count # create target value mapping
            count +=1
            fldr_data = pd.DataFrame(read_folder(fld_path))
            fldr_data['target'] = CATEGORY_MAPPING[x] # append target value
            DATA = DATA.append(fldr_data, ignore_index=True)

    # shuffle and partition data. roughly 80% partition
    DATA = shuffle(DATA)
    n = int(len(DATA)*0.8)
    TRAIN_SET, TEST_SET = DATA[:n], DATA[n:]

    print "Training examples = ", len(TRAIN_SET)
    print "Testing examples = ", len(TEST_SET)

    # write data
    TRAIN_SET.to_csv("training.csv", index=False,header=False)
    TEST_SET.to_csv("test.csv", index=False,header=False)

    # print target mapping to text file
    print "Target Mapping:"
    for k, v in CATEGORY_MAPPING.iteritems():
        print k, " = ", v

def main():
    parse_data()

if __name__ == '__main__':
    main()
