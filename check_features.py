import sys,os

MANIFEST = "MANIFEST.txt"
BASE = "./data/"

complete_features = {}

def read_folder(fn):
    all_features = []
    folder_mani = fn + MANIFEST
    with open(folder_mani,"r") as fm:
        fm.next()
        for line in fm:
            if 'annotation' not in line:
                filename = fn+line.split()[1]
                features = []
                with open(filename,"r") as miRNA_data:
                    miRNA_data.next()
                    for data_line in miRNA_data:
                        features.append(data_line.split()[0])
        all_features.append(features)
    return all_features

def check_features(fn):
    '''
    Checks for exact match (content + order)
    '''
    all_features = read_folder(fn)
    first_feature = all_features[0]
    for feature in all_features[1:]:
        if feature != first_feature:
            return False
    complete_features[fn] = first_feature
    return True

def check_all(base=BASE):
    all_dir  = []
    for x in os.listdir(base):
        if (x!= ".DS_Store"):
            all_dir.append(base+x+"/")
    for dirc in all_dir:
        if (not check_features(dirc)):
            print "Mismatch in", dirc

    folder_names = complete_features.keys()
    first_folder_features = complete_features[folder_names[0]]
    flag = True
    for fn in folder_names[1:]:
        if complete_features[fn] != first_folder_features:
            print "Mismatch between", fn, "and",folder_names[0]
            flag = False
    return flag

def main():
    if (check_all()):
        print "YAY!"
    else:
        print "Shucks!"

if __name__ == '__main__':
    main()
