from flask import Flask, make_response, jsonify, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import numpy as np
import pandas as pd

app = Flask(__name__)

rows = []


@app.route('/getDB')
def emplace():
    prod_name = ''

    prod_name = request.args.get('name')

    res = requests.get('http://43.201.114.241:8080/userinfo/product/')

    rows = res.json()

    product_name = []
    ingredients_list = []
    calories = []
    price = []
    amount = []
    carbohydrate = []
    cholesterol = []
    company = []
    cooking_type = []
    fat = []
    id = []
    primary_type = []
    product_category = []
    product_image = []
    protein = []
    sat_fat = []
    secondary_type = []
    serving_size = []
    sodium = []
    specific = []
    sugar = []
    trans_fat = []
    vegan_option = []

    for i in range(len(rows)):
        product_name.append(rows[i]["product_name"])
        ingredients_list.append(rows[i]["ingredient"])
        calories.append(rows[i]["calory"])
        price.append(rows[i]["price"])
        amount.append(rows[i]["amount"])
        carbohydrate.append(rows[i]["carbohydrate"])
        cholesterol.append(rows[i]["cholesterol"])
        vegan_option.append(rows[i]["vegan_option"])
        trans_fat.append(rows[i]["trans_fat"])
        sugar.append(rows[i]["sugar"])
        company.append(rows[i]["company"])
        cooking_type.append(rows[i]["cooking_type"])
        fat.append(rows[i]["fat"])
        id.append(rows[i]["id"])
        primary_type.append(rows[i]["primary_type"])
        product_category.append(rows[i]["product_category"])
        product_image.append(rows[i]["product_image"])
        protein.append(rows[i]["protein"])
        sat_fat.append(rows[i]["sat_fat"])
        secondary_type.append(rows[i]["secondary_type"])
        serving_size.append(rows[i]["serving_size"])
        sodium.append(rows[i]["sodium"])
        specific.append(rows[i]["specific"])

    data = pd.DataFrame({"product_name": product_name,
                        "ingredients_list": ingredients_list,
                         "calories": calories,
                         "price": price,
                         "amount": amount,
                         "carbohydrate": carbohydrate,
                         "cholesterol": cholesterol,
                         "company": company,
                         "cooking_type": cooking_type,
                         "fat": fat,
                         "id": id,
                         "primary_type": primary_type,
                         "product_category": product_category,
                         "product_image": product_image,
                         "protein": protein,
                         "sat_fat": sat_fat,
                         "secondary_type": secondary_type,
                         "serving_size": serving_size,
                         "sodium": sodium,
                         "specific": specific,
                         "sugar": sugar,
                         "trans_fat": trans_fat,
                         "vegan_option": vegan_option})

    cnt_vector = CountVectorizer(ngram_range=(1, 3))
    vector_categ = cnt_vector.fit_transform(data['ingredients_list'])
    categ_sim = cosine_similarity(
        vector_categ, vector_categ).argsort()[:, ::-1]

    target_idx = data[data['product_name'] == prod_name].index.values
    target_idx = target_idx[:1]
    sim_idx = categ_sim[target_idx, :].reshape(-1)
    sim_idx = sim_idx[sim_idx != target_idx]

    result = data.iloc[sim_idx].sort_values('price', ascending=False)
    data = data.iloc[target_idx[0]]
    result = result[(result['price'] > (data.price*0.9)) &
                    (result['price'] < (data.price*1.1))]
    result = result[(result['calories'] > (data.calories*0.9)) &
                    (result['calories'] < (data.calories*1.1))]

    result = result.to_json(orient='records', force_ascii=False)

    return make_response(jsonify(result), 200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
