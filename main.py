import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
import streamlit as st

# Load dataset
def load_data():
    data = pd.read_csv('laptop-recomend
/laptop_data.csv')
    return data

# Preprocessing data
def preprocess_data(data):
    data = data[['name', 'user rating', 'Price', 'Processor', 'RAM', 'Graphic Processor']]
    data['user rating'] = data['user rating'].fillna(data['user rating'].mean())
    return data

# Build similarity matrix
def calculate_similarity(data):
    pivot_table = data.pivot_table(index='name', columns='user rating', values='Price', fill_value=0)
    similarity_matrix = cosine_similarity(pivot_table)
    return similarity_matrix, pivot_table

# Recommend laptops
def recommend_laptops(laptop_name, similarity_matrix, pivot_table):
    index = pivot_table.index.get_loc(laptop_name)
    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in similarity_scores[1:6]]
    return pivot_table.index[recommended_indices]

# Streamlit interface
def main():
    st.title("Sistem Rekomendasi Laptop")
    data = load_data()
    data = preprocess_data(data)
    similarity_matrix, pivot_table = calculate_similarity(data)

    st.write("### Data Laptop")
    st.dataframe(data.head())

    laptop_name = st.selectbox("Pilih Laptop:", pivot_table.index)

    if st.button("Rekomendasikan"):
        recommendations = recommend_laptops(laptop_name, similarity_matrix, pivot_table)
        st.write("### Rekomendasi Laptop")
        for rec in recommendations:
            st.write(rec)

if __name__ == '__main__':
    main()
