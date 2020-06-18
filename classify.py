"""Jala Alamin
    CS 440
    PS7
    I'm using my free late day for this assignment"""

import sys
import os
import argparse
import csv
import collections
import operator
import math
import time
import random
from prettytable import PrettyTable

"""mine explanation: I chose to improve nb. For this function I made a dictionary of the most discriminatory terms
    by finding the difference value between each class document frequency, then chose the top 
    5000 words. At 5000 words the difference is around 9. At 5000 words the accuracy of the testing data is extremely 
     close to using the full dictionary, however the training data results are predictibly less accurate which is show 
     below. Also, there is around a 1 second improvement considering nb is optimized.
     
     Jalas-MacBook-Pro:ps7 jalaalamin$ python3 classify.py farm-ads-train farm-ads-test mine
                  Training Data Results
+-----------------+-------------------+--------------------+
|                 | Class 1 Predicted | Class -1 Predicted |
+-----------------+-------------------+--------------------+
| Class  1 Actual |     TP = 1236     |      FN = 324      |
| Class -1 Actual |      FP = 211     |     TN = 1173      |
+-----------------+-------------------+--------------------+
Accuracy:0.8182744565217391
                  Testing Data Results
+-----------------+-------------------+--------------------+
|                 | Class 1 Predicted | Class -1 Predicted |
+-----------------+-------------------+--------------------+
| Class  1 Actual |      TP = 509     |      FN = 141      |
| Class -1 Actual |      FP = 110     |      TN = 439      |
+-----------------+-------------------+--------------------+
Accuracy:0.7906588824020017
time for mine: 7.418712854385376


Jalas-MacBook-Pro:ps7 jalaalamin$ python3 classify.py farm-ads-train farm-ads-test nb
                  Training Data Results
+-----------------+-------------------+--------------------+
|                 | Class 1 Predicted | Class -1 Predicted |
+-----------------+-------------------+--------------------+
| Class  1 Actual |     TP = 1326     |      FN = 234      |
| Class -1 Actual |      FP = 132     |     TN = 1252      |
+-----------------+-------------------+--------------------+
Accuracy:0.8756793478260869
                  Testing Data Results
+-----------------+-------------------+--------------------+
|                 | Class 1 Predicted | Class -1 Predicted |
+-----------------+-------------------+--------------------+
| Class  1 Actual |      TP = 525     |      FN = 125      |
| Class -1 Actual |      FP = 91      |      TN = 458      |
+-----------------+-------------------+--------------------+
Accuracy:0.8198498748957465
time for nb: 8.233803033828735"""


def nb(docs, testdata, flag):
    tp, fp, tn, fn, c1, c2 = 0, 0, 0, 0, 0, 0
    pwcSum = [0, 0]

    for doc in docs:
        if doc[0] == '1':
            c1 += 1
        else:
            c2 += 1
    total = [c1, c2]
    if flag == 0:
        dict = df(docs, 0)
    else:
        dict = flag  # docs is a smaller dict

    # get a starting sum of the probability that no words in the dict are in the doc
    # then adjust for each doc to account for words that are there
    for w in dict.keys():
        pwcSum[0] += math.log(1 - ((dict[w][0] + 1) / (total[0] + 2)))
        pwcSum[1] += math.log(1 - ((dict[w][1] + 1) / (total[1] + 2)))

    savesum0 = pwcSum[0]
    savesum1 = pwcSum[1]
    for doc in docs:
        pwcSum = [savesum0, savesum1]
        seen = {}
        for word in doc:
            seen[word] = True
        if nb_helper(doc, docs, dict, c1, c2, total, seen, pwcSum) == 1:
            if doc[0] == '1':
                tp += 1
            else:
                fp += 1
        else:
            if doc[0] == '1':
                fn += 1
            else:
                tn += 1
    print("                  Training Data Results")
    printTable(tp, fp, tn, fn)

    tp, fp, tn, fn = 0, 0, 0, 0
    for doc in testdata:
        pwcSum = [savesum0, savesum1]
        seen = {}
        for word in doc[1:]:
            seen[word] = True
        if nb_helper(doc, docs, dict, c1, c2, total, seen, pwcSum) == 1:
            if doc[0] == '1':
                tp += 1
            else:
                fp += 1
        else:
            if doc[0] == '1':
                fn += 1
            else:
                tn += 1
    print("                  Testing Data Results")
    printTable(tp, fp, tn, fn)


def nb_helper(curdoc, docs, dict, c1, c2, total, seen, pwcSum):
    pc1, pc2, pdc1, pdc2, pcd1, pcd2 = 0, 0, 0, 0, 0, 0

    pc1 = c1 / (c1 + c2)
    pc2 = c2 / (c1 + c2)

    pdc1 = pdc(curdoc, dict, 0, total, seen, pwcSum)
    pdc2 = pdc(curdoc, dict, 1, total, seen, pwcSum)

    pcd1 = math.log(pc1) + pdc1
    pcd2 = math.log(pc2) + pdc2

    if pcd1 > pcd2:
        return 1
    else:
        return 2


    """second parameter is the index of the class"""
def pdc(curdoc, dict, c, total, seen, pwcSum):
    pdc, pwc = 0, 0
    for w in curdoc[1:]:
        if w in dict:
            pwcSum[c] -= math.log(
                (1 - (dict[w][c] + 1) / (total[c] + 2)))  # subtract the assumption that the word wasnt there
            pwcSum[c] += math.log((dict[w][c] + 1) / (total[c] + 2))  # add the probability

    return pwcSum[c]


"""this function classifies if the doc is in class 1 or -1"""
def mnb(docs, testdata):
    tp, fp, tn, fn, total, class1, class2 = 0, 0, 0, 0, 0, 0, 0

    bag = tf(docs, 0)
    # for every doc in docs calculate c*, then evaluate
    bag_count = [0, 0]
    bag_count1 = 0
    bag_count2 = 0
    # for w in list(bag.values()):
    # bag_count += w
    for k in bag:
        bag_count1 += bag[k][0]
        bag_count2 += bag[k][1]
    bag_count = [bag_count1, bag_count2]
    for doc in docs:
        total += 1
        if doc[0] == '1':
            class1 += 1
        else:
            class2 += 1
    for doc in docs:
        if mnb_helper1(class1, class2, bag, doc, bag_count) == 1:
            if doc[0] == '1':
                tp += 1
            else:
                fp += 1
        else:  # predict class -1
            if doc[0] == '-1':
                tn += 1
            else:
                fn += 1
    print("                  Training Data Results")
    printTable(tp, fp, tn, fn)
    tp, fp, tn, fn, total = 0, 0, 0, 0, 0
    print("")
    for doc in testdata:
        total += 1
        if mnb_helper1(class1, class2, bag, doc, bag_count) == 1:
            if doc[0] == '1':
                tp += 1
            else:
                fp += 1
        else:  # predict class -1
            if doc[0] == '-1':
                tn += 1
            else:
                fn += 1

    print("                  Testing Data Results")
    printTable(tp, fp, tn, fn)


def printTable(tp, fp, tn, fn):
    tpstr = 'TP = ' + str(tp)
    tnstr = 'TN = ' + str(tn)
    fpstr = 'FP = ' + str(fp)
    fnstr = 'FN = ' + str(fn)

    t = PrettyTable(['        ', 'Class 1 Predicted', 'Class -1 Predicted'])
    t.add_row(['Class  1 Actual', tpstr, fnstr])
    t.add_row(['Class -1 Actual', fpstr, tnstr])
    print(t)
    acc = (tp + tn) / (tp + tn + fp + fn)
    print("Accuracy:" + str(acc))


def mnb_helper1(class1, class2, bag, curdoc, bag_count):
    p_of_c1 = class1 / (class1 + class2)
    p_of_c2 = class2 / (class1 + class2)
    p_of_d_c1 = mnb_helper2(bag, curdoc, 0, bag_count)
    p_of_d_c2 = mnb_helper2(bag, curdoc, 1, bag_count)
    if p_of_d_c1 == 0 or p_of_d_c2 == 0 or p_of_c1 == 0 or p_of_c2 == 0:
        print("log error")
        return 0

    p_c1_d = p_of_d_c1 + math.log(p_of_c1)  # p_of_d_c1 * p_of_c1
    p_c2_d = p_of_d_c2 + math.log(p_of_c2)  # p_of_d_c2 * p_of_c2

    if p_c1_d > p_c2_d:
        return 1
    else:
        return 2

    """this function computes P(d|c)
    for every term in bag
    p(w|c) = # times w appears among all words in c/ total # of words in bag
    n = # of times w is seen in doc
    """
def mnb_helper2(bag, curdoc, i, bag_count):
    p_of_d_c = 0
    nlist = collections.Counter()
    nlist.update(curdoc[1:])
    nlist = dict(nlist)
    for word in curdoc[1:]:  # equivalent to for all words in dict because n=0 if word not in doc...
        if word in bag:
            p_of_w_c = pwc(word, i, bag, bag_count, curdoc)
            n = nlist[word]  # getn(word, curdoc)
            p_of_d_c += (math.log(p_of_w_c) * n)

    return p_of_d_c

"""c indicates class index 0 or 1"""
def pwc(word, c, bag, bag_count, curdoc):
    wc = 0
    wc = bag[word][c]
    wc += 1  # smoothing to avoid 0 probability when tf=0 for a word
    return wc / (bag_count[c] + 1)


def priors(docs, testdata):
    class1 = 0
    class2 = 0
    zeroRclass = 0
    for doc in docs:
        if doc[0] == '1':
            class1 += 1
        else:
            class2 += 1
    print("                  Training Data Results")

    if class1 > class2:
        zeroRclass = '1'
        printTable(class1,class2,0,0)
    else:
        zeroRclass = '-1'
        printTable(0,0,class1, class2)
    class1 = 0
    class2 = 0
    print("                  Testing Data Results")
    for doc in testdata:
        if doc[0] == '1':
            class1 += 1
        else:
            class2 += 1
    if zeroRclass == '1':
        printTable(class1, class2, 0, 0)
    else:
        printTable(0, 0, class1, class2)

def tfgrep_helper(tfdict, docs, maxDiffKey):
    actualClassOne = 0  # num of examples in class 1
    actualClassNeg = 0  # num of examples in class -1
    tp = 0
    fp = 0
    tn = 0
    total = 0
    # print(maxDiff)
    for doc in docs:
        total += 1
        if doc[0] == '1':
            actualClassOne += 1
        else:
            actualClassNeg += 1
        for word in doc:
            if word == maxDiffKey:
                if doc[0] == '1':
                    tp += 1
                elif doc[0] == '-1':
                    fp += 1
                break
    tpstr = 'TP = ' + str(tp)
    tn = 'TN = ' + str(actualClassNeg - fp)
    fpstr = 'FP = ' + str(fp)
    fnstr = 'FN = ' + str(actualClassOne - tp)
    num_of_docs = actualClassNeg + actualClassOne
    t = PrettyTable(['        ', 'Class 1 Predicted', 'Class -1 Predicted'])
    t.add_row(['Class  1 Actual', tpstr, fnstr])
    t.add_row(['Class -1 Actual', fpstr, tn])
    print(t)
    print(num_of_docs)
    tn = actualClassNeg - fp
    fn = actualClassOne - tp
    acc = (tp+tn)/(tp+tn+fp+fn)
    print("Accuracy:" + str(acc))


def tfgrep(tfdict, docs, docs2):
    maxDiff = 0
    maxDiffKey = ''
    for key in (tfdict.keys()):
        diff = tfdict[key][0] - tfdict[key][1]
        if diff < 0: diff *= -1
        if (diff > maxDiff):
            maxDiff = diff
            maxDiffKey = key
    print("disciminating term is %s" % maxDiffKey)
    print("                  Training Data Results")
    tfgrep_helper(tfdict, docs, maxDiffKey)
    with open(docs2) as f:
        data = f.read().splitlines()

    data = [list(elem.split()) for elem in data]
    print("                  Testing Data Results")
    tfgrep_helper(tfdict, data, maxDiffKey)


"""flag to print top five words"""


def tf(docs, flag):
    i = 0
    posOne = collections.Counter()
    negOne = collections.Counter()
    for doc in docs:
        if doc[0] == '1':
            posOne.update(doc[1:])
        else:
            negOne.update(doc[1:])

    posOne = dict(posOne)
    negOne = dict(negOne)
    result = {}
    for key in (posOne.keys()):
        result.setdefault(key, []).append(posOne[key])
    for key in (negOne.keys()):
        if key not in result: result.setdefault(key, []).append(0)
        result.setdefault(key, []).append(negOne[key])
    for key in (result.keys()):
        if len(result[key]) < 2:
            result.setdefault(key, []).append(0)
    with open('tf.csv', 'w+', newline='') as csvfile:
        i = 0
        w = csv.DictWriter(csvfile, fieldnames=['Term', 'Class 1 Frequency', 'Class -1 Frequency', ], delimiter=',')
        w.writeheader()
        w = csv.writer(csvfile)
        for key, v in result.items():
            w.writerow([key, v[0], v[1]])
        posOne_sorted = sorted(posOne.items(), key=operator.itemgetter(1), reverse=True)
        negOne_sorted = sorted(negOne.items(), key=operator.itemgetter(1), reverse=True)
        if flag == 1:
            print("Class 1: Top 5")
            print('---------------')
            for w in posOne_sorted[0:5]:
                print(w[0], w[1])
            print("")
            print("Class -1: Top 5")
            print('---------------')
            for w in negOne_sorted[0:5]:
                print(w[0], w[1])

        return result


"""returns a dict with terms and # of docs they appear in """

def df(docs, flag):
    result = {}
    c1List = {}
    c2List = {}
    saw = []
    # add all words to dict
    for doc in docs:
        for w in doc[1:]:
            result[w] = [0, 0]
    for doc in docs:
        saw = []
        for w in doc[1:]:
            if doc[0] == '1':
                if w not in saw:
                    result[w][0] += 1
                    saw.append(w)
            else:
                if w not in saw:
                    result[w][1] += 1
                    saw.append(w)
    with open('df.csv', 'w+', newline='') as csvfile:
        w = csv.DictWriter(csvfile, fieldnames=['Term', 'Class 1 Frequency', 'Class -1 Frequency', ], delimiter=',')
        w.writeheader()
        w = csv.writer(csvfile)
        for key, v in result.items():
            w.writerow([key, v[0], v[1]])
    for k, v in result.items():
        c1List[k] = v[0]
        c2List[k] = v[1]
    c1List = sorted(c1List.items(), key=operator.itemgetter(1), reverse=True)
    c2List = sorted(c2List.items(), key=operator.itemgetter(1), reverse=True)
    if flag == 1:
        print("Class 1: Top 5")
        print('---------------')
        for w in c1List[0:5]:
            print(w[0], w[1])
        print("")
        print("Class -1: Top 5")
        print('---------------')
        for w in c2List[0:5]:
            print(w[0], w[1])
    return result


def mine(docs, testdata):
    dict = df(docs, 0)
    disc_dict = {}
    for k in dict:
        val = abs(dict[k][0] - dict[k][1])
        disc_dict[k] = val
    disc_list = sorted(disc_dict.items(), key=lambda kv: kv[1], reverse=True)

    # grab first 50000 most discriminating terms throw out rest of words
    disc_list = disc_list[0:5000]
    disc_dict = {term[0]: term[1] for term in disc_list}

    # disc_dict needs to have vals from both classes!
    newdocs = {k: dict[k] for k in dict if k in disc_dict}
    nb(docs, testdata, newdocs)

def perceptron(docs,testdata):
    tp,tn,fn,fp = 0,0,0,0
    weights = {} #dict of the weight of each term in features
    #initialize weights to random small values...
    for doc in docs:
        for term in doc[1:]:
            weights[term] = random.uniform(0,1)

    wb=random.uniform(0,1)
    b=1
    g=0
    err =0
    alpha = 0.8
    random.shuffle(docs)
    for x in range(10):
        tp, tn, fn, fp = 0, 0, 0, 0
        for doc in docs:
            features = {}
            inSum = 0
            for f in doc[1:]:
                features[f] = 1 #df feature vector
            for f in doc[1:]:
                inSum += weights[f] * features[f]
            inSum += wb*b
            if inSum > 0 :
                g=1
            else: g=0
            #calculate error err = actual - predicted class 1 -> 1, class -1 -> 0
            if doc[0] == '1' :
                if g==1:
                    err = 0
                    tp +=1
                else:
                    err = 1
                    fn += 1
            else: #class is "0"
                if g==1:
                    err = -1
                    fp += 1
                else:
                    err = 0
                    tn += 1
            #calculate modified weights
            if err != 0:
                for term in features:
                    weights[term] = weights[term] + (alpha*err* features[term])
                wb = wb + (alpha*err*b)

    print("                  Training Data Results(at iteration #10")
    printTable(tp,fp,tn,fn)
    tp, tn, fn, fp = 0, 0, 0, 0
    for doc in testdata:
        features = {}
        inSum = 0
        for f in doc[1:]:
            features[f] = 1 #df feature vector
        for f in doc[1:]:
            if f in weights:
                inSum += weights[f] * features[f]
        if inSum > 0 :
            g=1
        else: g=0
        #calculate error err = actual - predicted class 1 -> 1, class -1 -> 0
        if doc[0] == '1' :
            if g==1:
                tp +=1
            else:
                fn += 1
        else: #class is "0"
            if g==1:
                fp += 1
            else:
                tn += 1
    print("                  Testing Data Results")
    printTable(tp,fp,tn,fn)
if __name__ == "__main__":
    tfdict = {}
    if len(sys.argv) > 0:
        with open(sys.argv[1]) as f:
            data = f.read().splitlines()

    data = [list(elem.split()) for elem in data]
    if len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            testdata = f.read().splitlines()
    testdata = [list(elem.split()) for elem in testdata]

    if len(sys.argv) == 4:
        if (sys.argv[3] == 'tf'):
            tf(data, 1)
        elif (sys.argv[3] == 'tfgrep'):
            tfdict = tf(data, 0)
            tfgrep(tfdict, data, sys.argv[2])
        elif (sys.argv[3] == 'priors'):
            priors(data, testdata)
        elif (sys.argv[3] == 'mnb'):
            mnb(data, testdata)
        elif (sys.argv[3] == 'df'):
            df(data, 1)
        elif (sys.argv[3] == 'nb'):
            start = time.time()
            nb(data, testdata, 0)
            end = time.time()
            print("time for nb: " + str(end - start))
        elif (sys.argv[3] == 'mine'):
            start = time.time()
            mine(data, testdata)
            end = time.time()
            print("time for mine: " + str(end - start))
        elif (sys.argv[3] == 'perceptron'):
            perceptron(data,testdata)
        else:
            print("Error: function is not valid")
            print("Valid functions are:")
            print("tf: Determines term frequencies for each word in the training set")
            print("tfgrep: Classifies a document based on the most discriminating term")
            print("priors: Classifies a document using O-R (guesses majority class every time)")
            print("mnb: Classifies a document using the Multinomial Naive Bayes Model")
            print("df: Determines the document frequency for each term in the training set")
            print("nb: Classifies a document using the multivariate Bernoulli model")
            print("mine: Classifies a document using the multivariate Bernoulli model given a smaller dictionary")
            print("perceptron: Classifies a document using perceptron learning with a hard threshold")
    else:
        print("Error: Number of arguments")
        print("please input 1) a file of training data 2) a file of testing data 3) a function")
        print("Valid functions are:")
        print("tf: Determines term frequencies for each word in the training set")
        print("tfgrep: Classifies a document based on the most discriminating term")
        print("priors: Classifies a document using O-R (guesses majority class every time)")
        print("mnb: Classifies a document using the Multinomial Naive Bayes Model")
        print("df: Determines the document frequency for each term in the training set")
        print("nb: Classifies a document using the multivariate Bernoulli model")
        print("mine: Classifies a document using the multivariate Bernoulli model given a smaller dictionary")