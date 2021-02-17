from pyspark import SparkContext, SparkConf
import sys
import json
from itertools import combinations
import math
import time

# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

train_file = 'train_review.json'
model_file = 'model.model'

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


def pearson_cor(iterator, busi_dict):
    for b_id1, b_id2 in iterator:
        b1_dict = busi_dict[b_id1]
        b2_dict = busi_dict[b_id2]
        co_rated = set(b1_dict.keys()) & set(b2_dict.keys())
        r1_corated_total = 0
        r2_corated_total = 0
        for key in co_rated:
            r1_corated_total += b1_dict[key]
            r2_corated_total += b2_dict[key]
        r1_ave = r1_corated_total / len(co_rated)
        r2_ave = r2_corated_total / len(co_rated)
        numerator = 0
        dnmnt_b1 = 0
        dnmnt_b2 = 0
        for cor in co_rated:
            numerator += (b1_dict[cor] - r1_ave) * (b2_dict[cor] - r2_ave)
            dnmnt_b1 += pow((b1_dict[cor] - r1_ave), 2)
            dnmnt_b2 += pow((b2_dict[cor] - r2_ave), 2)

        denominator = math.sqrt(dnmnt_b1 * dnmnt_b2)
        if denominator != 0:
            pearson = numerator / denominator
            if 0 < pearson <= 1:
                yield (b_id1, b_id2, pearson)


def main(sc):
    review = sc.textFile(train_file) \
        .map(lambda s: json.loads(s)) \
        .persist()

    business_rate = review.map(lambda s: ((s['business_id'], s['user_id']), s['stars'])) \
        .aggregateByKey((0, 0), lambda u, v: (u[0] + v, u[1] + 1), lambda U, V: (U[0] + V[0], U[1] + V[1])) \
        .mapValues(lambda v: v[0] / v[1]) \
        .map(lambda s: (s[0][0], (s[0][1], s[1]))) \
        .groupByKey() \
        .mapValues(list) \
        .map(lambda s: (s[0], dict(s[1]))) \
        .collectAsMap()

    business_pairs = review.map(lambda s: (s['user_id'], s['business_id'])) \
        .distinct() \
        .groupByKey() \
        .mapValues(list) \
        .filter(lambda s: len(s[1]) > 1) \
        .flatMap(lambda s: combinations(sorted(s[1]), 2)) \
        .map(lambda s: ((s[0], s[1]), 1)) \
        .reduceByKey(lambda u, v: u + v) \
        .filter(lambda s: s[1] >= 3) \
        .map(lambda s: s[0]) \
        .mapPartitions(lambda s: pearson_cor(s, business_rate)) \
        .map(lambda s: dict((('b1', s[0]), ('b2', s[1]), ('sim', s[2])))) \
        .collect()

    with open(model_file, 'w') as f:
        for piece in business_pairs:
            f.write(json.dumps(piece) + '\n')

    print('Duration: ', time.time() - start_time)

if __name__ == '__main__':
    start_time = time.time()
    sc = init_spark()
    main(sc)