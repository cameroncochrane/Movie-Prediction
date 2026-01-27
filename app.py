from flask import Flask, render_template, request, jsonify
from functions import *

app = Flask(__name__)

# Load the movie catalogue and build the TF-IDF model once when the server starts
catalogue = load_catalogue_with_soup()
recommender = ContentRecommender(catalogue)


@app.route("/", methods=["GET"])
def index():
    # The dropdown options are loaded dynamically via /search,
    # so we don’t need to pass all 5000 titles into the template.
    return render_template("index.html")


@app.route("/search", methods=["GET"])
def search():
    """
    Called by Select2 as the user types.

    Select2 expects JSON in the form:
      {"results": [{"id": <value>, "text": <label>}, ...]}

    We do a simple "contains" match on titles.
    """
    q = (request.args.get("q") or "").strip().lower()

    # Require some typing before we search
    if len(q) < 2:
        return jsonify({"results": []})

    # Case-insensitive contains match
    mask = catalogue["title"].str.lower().str.contains(q, na=False)
    hits = catalogue.loc[mask, ["id", "title", "release_date"]].head(25)

    # Make labels a bit clearer by including year if available
    results = []
    for _, r in hits.iterrows():
        year = ""
        rd = r.get("release_date") or ""
        if isinstance(rd, str) and len(rd) >= 4:
            year = rd[:4]
        label = f"{r['title']} ({year})" if year else r["title"]

        results.append({"id": int(r["id"]), "text": label})

    return jsonify({"results": results})


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Called when the user submits the form after selecting a movie.
    """
    movie_id = int(request.form.get("movie_id"))
    top_n = int(request.form.get("top_n", 10))

    selected_row = catalogue.loc[catalogue["id"] == movie_id].iloc[0].to_dict()
    recs = recommender.recommend_by_id(movie_id=movie_id, top_n=top_n)

    return render_template("results.html", selected=selected_row, recs=recs)


if __name__ == "__main__":
    app.run(debug=True)