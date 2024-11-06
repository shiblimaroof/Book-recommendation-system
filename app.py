import pickle
import numpy as np
import streamlit as st
import pandas as pd

# Load the required data
st.header('Books Recommendation System using Machine Learning')
model = pickle.load(open('artifacts/model.pkl', 'rb'))
books_name = pickle.load(open('artifacts/books_name.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/books_pivot.pkl', 'rb'))

def fetch_poster(suggestion):
    books_names = []
    ids_index = []
    poster_url = []

    # Iterate over the book IDs in the suggestions
    for book_id in suggestion:
        # Ensure the book_id is valid and within bounds
        if book_id < len(book_pivot):
            book_name = book_pivot.index[book_id]
            books_names.append(book_name)

    # Fetch the indices from final_rating using the book names
    for name in books_names:
        # Check if the title exists
        if name in final_rating['title'].values:
            ids = np.where(final_rating['title'].values == name)[0]
            if ids.size > 0:
                ids_index.append(ids[0])  # Add the first matching index

    # Retrieve the poster URLs using the indices
    for idx in ids_index:
        if idx in final_rating.index:  # Check if idx is a valid index
            url = final_rating.loc[idx, 'img_url']  # Use loc to get the URL
            poster_url.append(url)

    return poster_url

def recommend_book(book_name):
    book_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]  # Get the index of the selected book
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=10)

    # Fetch posters for the suggestions
    poster_url = fetch_poster(suggestion[0])  # Use the first array of suggestion

    # Build the list of recommended books
    for i in range(len(suggestion[0])):  # Use suggestion[0] for the actual suggestions
        if suggestion[0][i] < len(book_pivot):  # Ensure we don't exceed bounds
            books = book_pivot.index[suggestion[0][i]]  # Index with the suggestion values
            book_list.append(books)

    return book_list, poster_url

# Streamlit UI for selecting a book
selected_books = st.selectbox(
    "Type or select a book",
    books_name
)

# Show recommendations when the button is pressed
if st.button('Show Recommendation'):
    try:
        recommendation_books, poster_url = recommend_book(selected_books)

        # Create columns to display the recommended books and their posters
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                if i < len(recommendation_books):  # Ensure we don't go out of range
                    st.text(recommendation_books[i])
                    if i < len(poster_url):  # Check for poster URLs length
                        st.image(poster_url[i])
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")