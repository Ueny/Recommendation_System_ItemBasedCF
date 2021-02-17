from pyspark import SparkContext, SparkConf
import sys
import json
import time
import os

# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

train_file = 'boosted_review.json'
test_file = 'test_review.json'
model_file = 'boosted.model'
output_file = 'predict.predict'

N = 15


def init_spark():
    '''
    initialize the spark context
    :return: spark context
    '''

    conf = SparkConf() \
        .setAppName('inf553_recommendation_system') \
        .setMaster('local[*]')
    sc = SparkContext(conf=conf)
    return sc


def predict_itembased(data, rating_dict, biz_avg_dict):
    bid, uid, neighbors = data
    if (uid, bid) in rating_dict.keys():
        return (uid, bid, rating_dict[(uid, bid)])
    elif bid not in biz_avg_dict.keys():
        return (uid, bid, 3)
    numereator = 0
    denominator = 0
    n = 0
    for neighbor, sim in neighbors:
        if sim <= 0.1:
            break
        if (uid, neighbor) in rating_dict.keys():
            numereator += rating_dict[(uid, neighbor)] * sim
            denominator += abs(sim)
            n += 1
            if n >= N:
                break
    print(n)
    if denominator == 0 or n <= 3:
        pred_res = biz_avg_dict[bid]
    else:
        pred_res = numereator/denominator
    return (uid, bid, pred_res)

def main(sc):
    # get ratings from the train data
    rating_dict = sc.textFile(train_file) \
        .map(lambda s: json.loads(s)) \
        .map(lambda s: ((s['user_id'], s['business_id']), s['stars'])) \
        .aggregateByKey((0, 0), lambda u, v: (u[0] + v, u[1] + 1), lambda U, V: (U[0] + V[0], U[1] + V[1])) \
        .mapValues(lambda v: v[0] / v[1]) \
        .persist()

    biz_avg_dict = rating_dict.map(lambda s: (s[0][1], s[1])) \
        .aggregateByKey((0, 0), lambda u, v: (u[0] + v, u[1] + 1), lambda U, V: (U[0] + V[0], U[1] + V[1])) \
        .mapValues(lambda v: v[0] / v[1]) \
        .collectAsMap()

    rating_dict = rating_dict.collectAsMap()

    # build models
    # deal with models into 2 parts
    model1 = sc.textFile(model_file) \
        .map(lambda s: json.loads(s)) \
        .map(lambda s: (s['b1'], (s['b2'], s['sim']))) \
        .persist()

    model = model1.map(lambda s: (s[1][0], (s[0], s[1][1]))) \
        .union(model1)\
        .persist()

    # load target pairs
    pred = sc.textFile(test_file) \
        .map(lambda s: json.loads(s)) \
        .map(lambda s: (s['business_id'], s['user_id'])) \
        .leftOuterJoin(model) \
        .map(lambda s: ((s[0], s[1][0]), s[1][1]) if s[1][1] is not None else ((s[0], s[1][0]), (None, -1))) \
        .groupByKey() \
        .mapValues(list) \
        .map(lambda s: (s[0][0], s[0][1], sorted(s[1], key=lambda x: x[1], reverse=True))) \
        .map(lambda s: predict_itembased(s, rating_dict, biz_avg_dict)) \
        .map(lambda s: dict({'user_id': s[0], 'business_id': s[1], 'stars': round(s[2])})) \
        .collect()

    with open(output_file, 'w') as f:
        for piece in pred:
            f.write(json.dumps(piece) + '\n')
    print('Duration: ', time.time() - start_time)


if __name__ == '__main__':
    start_time = time.time()
    sc = init_spark()
    main(sc)