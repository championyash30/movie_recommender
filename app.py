from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import os  # for getting the PORT from environment

app = Flask(__name__)

# Load data
movies = pickle.load(open("movies.pkl", "rb"))  # list of dicts with title + tmdb_id

# Load compressed similarity matrix
# (your file was saved without a key, so NumPy defaulted to "arr_0")
similarity_uint8 = np.load("similarity_final.npz")["arr_0"]
similarity = similarity_uint8.astype(np.float32) / 255.0  # scale back to [0, 1]

TMDB_API_KEY = "dc837d8573a26d40aae28bd55f7b8a16"

def get_poster_url(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return "https://via.placeholder.com/500x750?text=No+Image"

def get_overview(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US"
    data = requests.get(url).json()
    return data.get('overview', 'No description available.')

def recommend(movie_title):
    index = next((i for i, m in enumerate(movies) if m['title'].lower() == movie_title.lower()), None)
    if index is None:
        return []

    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recs = []
    for i in distances[1:6]:  # only 5 recommendations
        recs.append({
            "title": movies[i[0]]['title'],
            "poster": get_poster_url(movies[i[0]]['tmdb_id']),
            "tmdb_id": movies[i[0]]['tmdb_id'],
            "overview": get_overview(movies[i[0]]['tmdb_id'])
        })
    return recs

@app.route("/", methods=["GET", "POST"])
def index():
    movie_names = [m['title'] for m in movies]
    recommended = []
    selected_movie = ""
    if request.method == "POST":
        selected_movie = request.form.get("movie")
        recommended = recommend(selected_movie)
    return render_template("index.html", movies=movie_names, recommended=recommended,
                           selected=selected_movie, TMDB_API_KEY=TMDB_API_KEY)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # use Render's PORT or default 5000
    app.run(host="0.0.0.0", port=port)
