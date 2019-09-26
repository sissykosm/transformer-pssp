import numpy as np

# calculating accuracy 
def get_acc(gt,pred):
    assert len(gt) == len(pred)
    correct = 0
    for i in range(len(gt)):
        if gt[i] == pred[i]:
            correct += 1
                
    return (1.0 * correct)/len(gt)

try:
    fp = open('./pred.txt')
    fp2 = open('./pssp-data/pss_test.txt')
    line = fp.readline()
    res = fp2.readline()
    acc_list = []
    i = 1
    while (line):
        text = line.upper().replace("", " ").rstrip().lstrip()
        res = res.rstrip().lstrip()
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



            