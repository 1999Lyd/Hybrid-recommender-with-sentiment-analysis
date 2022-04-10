# Hybrid-recommender-with-sentiment-analysis

![Hybrid recommender with reviews (2)](https://user-images.githubusercontent.com/31523376/162637097-2e48e5a8-32db-41f0-accc-c08319cff88a.jpg)

## Introduction

Imagine a scenario when you are planning to watch your favorite movie on weekend. You believe the movie deserves the companion of some tasty snacks. 

However, you just don’t want to waste your energy to think about what snack to eat, you want someone to recommend you the appropriate snacks that fit your taste…

Here comes the food recommendation system. It will automatically recommend the food you like, based on your past purchasing history, review and rating behaviors, and comes up with the top 5 recommendations that match your taste best.

## Data

This project makes use of the Amazon Fine Foods Review dataset. This consists of 10 data fields and ~568,00 records. This app utilizes the following records from the dataset: ProductId, UserId, Rate, and Preview.

## Getting Started

The model training process utilizes the flowchart below:

![Hybrid recommender with reviews (1)](https://user-images.githubusercontent.com/31523376/162636595-9e60e2bc-afd4-4ce7-a66d-bf79f4de43ad.jpg)


- Download the [dataset](https://duke.app.box.com/folder/160083268030?s=6ayc5muwnntphn89hq3bx3tgf273jjso) to the same directory of hybrid_model.py
- ```python hybrid_model.py``` see the cost path and get the trained model

## Demo

We built a recommendation engine using the neural network hybrid model. Given an amazon user, we recommend 5 related products based on products they’ve purchased previously and sentiment from their reviews. For the demo we compared the user’s top rated item with the recommendations.

Using the box folder located [here](https://duke.app.box.com/folder/160083268030?s=6ayc5muwnntphn89hq3bx3tgf273jjso), download the models and data in the following directories:

- ```./models/model.pt```
- ```./models/product_encoder.pickle```
- ```./models/user_encoder.pickle```
- ```./models/tfidf_reg.pickle```
- ```./models/tfidf.pickle```
- - ```./data/processed/app_data_test.csv```

Once the models are in the right directories, to run the app, use the following command to run the app:

```
streamlit run app.py
```
