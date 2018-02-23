import sys,os
import pandas as pd
from sklearn.utils import shuffle

MANIFEST = "MANIFEST.txt"
BASE = "./data/"

CATEGORY_MAPPING = {}

def read_row(full_path):
    row = []
    with  open(full_path,'r') as row_f:
        row_f.next()
        for line in row_f:
            row.append(line.split()[2])
    #print len(row)
    return row

def read_folder(fn):
    all_rows = []
    folder_mani = fn + MANIFEST
    with open(folder_mani,"r") as fm:
        fm.next()
        for line in fm:
            if 'annotation' not in line:
                filename = fn+line.split()[1]
                row = read_row(filename)
                all_rows.append(row)
    #print len(all_rows)
    return all_rows

def parse_data():
    count = 0
    DATA = pd.DataFrame()
    for x in os.listdir(BASE):
        if (".DS_Store" not in x):
            fld_path = BASE+x+"/"
            CATEGORY_MAPPING[x] = count
            count +=1
            fldr_data = pd.DataFrame(read_folder(fld_path))
            fldr_data['target'] = CATEGORY_MAPPING[x]
            DATA = DATA.append(fldr_data, ignore_index=True)

    DATA = shuffle(DATA)
    n = int(len(DATA)*0.8)
    TRAIN_SET, TEST_SET = DATA[:n], DATA[n:]
    TRAIN_SET.to_csv("training.csv", index=False,header=False)
    TEST_SET.to_csv("test.csv", index=False,header=False)

def main():
    parse_data()
if __name__ == '__main__':
    main()
