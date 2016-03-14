# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd


def similarity(x, y):
    nx = x / np.linalg.norm(x)
    ny = y / np.linalg.norm(y)
    return np.dot(nx, ny)


def compute_ranking(x, ys, names):
    res = []
    for n in xrange(len(ys)):
        y = ys[n]
        name = names[n]
        score = similarity(x, y)
        res.append((score, name))
    res.sort()
    res.reverse()
    return res


def main():
    data = pd.read_csv("caltech101_vggnet_fc7_features.csv")

    data = data.as_matrix()

    names = data[:,0]
    data = data[:,1:].astype(np.float64)
    N = len(names)

    index = np.random.permutation(N)
    names = names[index]
    data = data[index]

    K = 5
    for n in xrange(N):
        ranked_names = compute_ranking(data[n], data, names)
        print "Query: %s" % names[n]
        for k in xrange(1,K+1):
            print "Retrieved: %s [similarity = %f]" % (ranked_names[k][1], ranked_names[k][0])


if __name__ == "__main__":
    main()
