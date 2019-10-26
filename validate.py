import numpy as np
import argparse

# calculating accuracy 
def get_acc(gt,pred):
    assert len(gt) == len(pred)
    correct = 0
    for i in range(len(gt)):
        if gt[i] == pred[i]:
            correct += 1
                
    return (1.0 * correct)/len(gt)

try:
    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-txt', required=True,
                        help='Path to decoded text file')

    parser.add_argument('-test', help='Path to test labels file', default="./pssp-data/pss_test.txt")

    opt = parser.parse_args()
    fp = open(opt.txt)
    fp2 = open(opt.test)
    line = fp.readline()
    res = fp2.readline()
    acc_list = []
    i = 1
    while (line):
        text = line.upper().rstrip().lstrip()
        res = res.replace(" ", "").rstrip().lstrip()
        i = i + 1
        if (len(text) == len(res)):
            acc = get_acc(res, text)
            print(i, acc)
            acc_list.append(acc)
        else: 
            print("error")
        
        line = fp.readline()
        res = fp2.readline()

    accuracy = np.mean(acc_list)
    print("Accuracy is: ", accuracy)
        

    # do stuff here
finally:
    fp.close()
    fp2.close()



            