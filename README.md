# 🎬 Movie Recommendation System

A **content-based movie recommender** built with **Python, NLP and Machine Learning**. Pick a movie you like, and the app instantly suggests similar movies.

🔗 **Live Demo:** [Open the app](YOUR_STREAMLIT_APP_LINK)

---

## 📌 Overview

This project recommends movies by comparing their content (overview, genres, keywords, cast, etc.). Each movie is converted into a numerical vector, and **cosine similarity** finds the movies that are closest to the one you select.

## ✨ Features

- Content-based recommendations (no user login or ratings needed)
- Clean text processing using **stemming**
- Fast similarity search with **cosine similarity**
- Simple and interactive **Streamlit** web interface
- Deployed live on **Streamlit Community Cloud**

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| NLP | NLTK (Stemming), CountVectorizer |
| ML / Similarity | Scikit-learn (Cosine Similarity) |
| Web App | Streamlit |
| Deployment | Streamlit Community Cloud, GitHub |

## ⚙️ How It Works

1. **Data Cleaning:** Load the movie dataset and keep the important columns.
2. **Text Preprocessing:** Combine movie details into one text column (`tags`), convert to lowercase and apply **stemming**.
3. **Vectorization:** Convert the text into numbers using **CountVectorizer** (Bag-of-Words).
4. **Similarity Calculation:** Calculate **cosine similarity** between all movie vectors.
5. **Recommendation:** For the selected movie, return the top 5 most similar movies.
6. **Web App:** Show the results in a **Streamlit** interface.

## 📂 Project Structure

```
movie-recommendation-system/
│
├── app.py                 # Streamlit app
├── requirements.txt       # Dependencies
├── movies.pkl             # Processed movie data
├── similarity.pkl         # Cosine similarity matrix
├── notebook.ipynb         # Data processing and model building
└── README.md
```

> Update the file names above to match your repository.

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/sagardubeyokk/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

## 📸 Screenshots

Add 1–2 screenshots of your app here:

```
![App Screenshot](screenshots/app.png)
```

## 🔮 Future Improvements

- Show movie posters using the TMDB API
- Combine content-based and collaborative filtering
- Try TF-IDF or embeddings for better accuracy
- Add search filters (genre, year, rating)

## 👨‍💻 Author

**Sagar**
GitHub: [@sagardubeyokk](https://github.com/sagardubeyokk)

---

⭐ If you like this project, please give it a star!
