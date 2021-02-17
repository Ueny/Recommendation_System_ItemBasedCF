from sklearn.metrics import mean_squared_error
from math import sqrt
import json


my_file = 'predict.predict'
truth_file = 'test_review_ratings.json'

with open(my_file, 'r') as f1:
    my_res = f1.readlines()

my_dic = dict()
for line in my_res:
    line_js = json.loads(line)
    my_dic[(line_js['user_id'], line_js['business_id'])] = line_js['stars']


with open(truth_file, 'r') as f2:
    truth_res = f2.readlines()

true_dic = dict()
for line in truth_res:
    line_js = json.loads(line)
    true_dic[(line_js['user_id'], line_js['business_id'])] = line_js['stars']

pred = []
actual = []
for key, value in my_dic.items():
    pred.append(value)
    actual.append(true_dic[key])

rmse = sqrt(mean_squared_error(actual, pred))
print(rmse)