import sys
import os
import glob
import random
import re

if __name__ == "__main__":

    TRAIN_SAVE_PATH = "./train.txt"

    TEST_SAVE_PATH = "./test.txt"

    BASE_PATH = '/exdata/ylyg/final_crop/'
    SEPARATOR = ' '
    train_fh = open(TRAIN_SAVE_PATH, 'w')
    test_fh = open(TEST_SAVE_PATH, 'w')

    label = 0
    for root, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(root, subdirname)
            file_list = os.listdir(subject_path)
            file_list = [name for name in file_list if name.endswith('.png')]
            length = len(file_list)
            print("\nnums of", subdirname, ": ", length)
            random.shuffle(file_list)
            train_len = int(0.75 * length)
            for filename in file_list[:train_len]:
                train_fh.write(subdirname + '/' + filename + SEPARATOR + str(label) + "\n")
            for filename in file_list[train_len:]:
                test_fh.write(subdirname + '/' + filename + SEPARATOR + str(label) + "\n")
            #             for filename in os.listdir(subject_path):
            #                 abs_path = "%s/%s" % (subject_path, filename)
            #                 print "%s%s%d" % (abs_path, SEPARATOR, label)
            #                 fh.write(abs_path + SEPARATOR + str(label) + "\n")
            label = label + 1
    train_fh.close()
    test_fh.close()
