# Recommendation_System_ItemBasedCF

We generated the review data from the original Yelp datasets with some filters, such as the condition: “state” == “CA”. We randomly took 80% of the data for training, 10% of the data for testing, and 10% of the data as the blind dataset. We do not share the blind dataset.
You can access the files (a-e) under the fixed directory on the Vocareum: resource/asnlib/publicdata/
1. train_review.json
2. user.json – user metadata
3. business.json – business metadata, including locations, attributes, and categories
4. user_avg.json – containing the average stars for the users in the train dataset
5. business_avg.json – containing the average stars for the businesses in the train dataset. Besides, the Google Drive provides the above files (a-e) and the following testing files (f and g) https://drive.google.com/open?id=1ss6Tq-hxeRfyst8u-n8Tx8Ykn1jD8GB8 (USC email only)
6. test_review.json – containing only the target user and business pairs for the prediction task
7. test_review_ratings.json – containing the ground truth rating for the testing pairs

Training:
1. When I met that some users rated a business repeatedly, I took the average score as the true rating

2. Group the business by the user, and generate business pairs

3. Filter those business pairs which have been co-rated by more than 3 users

4. Calculate the Pearson Correlation of business pairs, and record it in the model file

Model:
1. Record the relational weight between two businesses
2. E.g., (“b1”: business_id, “b2”: business_id, “sim”: pearson correlation)

Predict:
1. Load training data (training file), model data(model file) and target data(test_review file) together
2. After merging, we get an assembled data structure: (target business_id, target user_id, [(neighbor business_id, sim), ...])
3. We are able to use at most 15 neighbor businesses with their weights (Pearson Correlation) to predict the target rating of the target user and business
- If target user and business already have a rating, then we use it directly
- If it is a new business (has not shown in training file), we give a 3 as the rating
- If the number of available neighbor businesses is less than 3, then we use the average rating of this business
