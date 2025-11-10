# vibe_matcher_app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ==================================
# CONFIGURATION
# ==================================
st.set_page_config(page_title="Vibe Matcher", layout="centered")

# 🔑  Configure Google Gemini API
GEMINI_API_KEY = "AIzaSyC3i4AGeF4esAFvIfPzqKKZsrHUD-XZxSU"
genai.configure(api_key=GEMINI_API_KEY)
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Apply dark mode chart style
import matplotlib
matplotlib.rcParams.update({
    "axes.facecolor": "#0e1117",
    "figure.facecolor": "#0e1117",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.titlecolor": "white"
})

# ==================================
# DATASET
# ==================================
products = [
    {"id": 1, "name": "Boho Dress", "desc": "Flowy maxi dress, earthy tones, embroidery — perfect for festival vibes.", "vibes": ["boho", "festival", "earthy"]},
    {"id": 2, "name": "Urban Bomber Jacket", "desc": "Cropped bomber with reflective trims — energetic urban chic.", "vibes": ["urban", "chic", "energetic"]},
    {"id": 3, "name": "Cozy Knit Sweater", "desc": "Oversized knit — cozy and warm.", "vibes": ["cozy", "casual", "warm"]},
    {"id": 4, "name": "Minimalist Slip Dress", "desc": "Satin slip dress in neutral tones — elegant and modern.", "vibes": ["minimal", "elegant", "modern"]},
    {"id": 5, "name": "Sporty Mesh Sneakers", "desc": "Lightweight sneakers with neon accents — playful and active.", "vibes": ["sporty", "active", "playful"]},
    {"id": 6, "name": "Tailored Blazer", "desc": "Structured blazer, sharp lapels — professional and confident city style.", "vibes": ["professional", "city", "confident"]},
    {"id": 7, "name": "Bohemian Kimono", "desc": "Patterned kimono with fringe — relaxed, artistic, and bohemian.", "vibes": ["boho", "artsy", "relaxed"]},
    {"id": 8, "name": "Futuristic Tech Vest", "desc": "Utility vest with modular pockets — edgy, tech, and experimental.", "vibes": ["tech", "experimental", "street"]},
]
df = pd.DataFrame(products)

# ==================================
# EMBEDDINGS
# ==================================
@st.cache_resource
def get_embeddings(texts):
    vectors = []
    for t in texts:
        r = genai.embed_content(model=EMBEDDING_MODEL, content=t)
        vectors.append(np.array(r["embedding"]))
    return np.vstack(vectors)
#Precompute embeddings for all products
prod_emb = get_embeddings(df["desc"].tolist())

def get_top_matches(query, k=3):
    start_time = time.perf_counter()
    q_emb = np.array(genai.embed_content(model=EMBEDDING_MODEL, content=query)["embedding"])
    sims = cosine_similarity([q_emb], prod_emb)[0]
    idx = np.argsort(sims)[::-1][:k]
    latency = (time.perf_counter() - start_time) * 1000
    return (
        [{"name": df.iloc[i]["name"], "desc": df.iloc[i]["desc"], "vibes": df.iloc[i]["vibes"], "score": float(sims[i])} for i in idx],
        latency
    )

# ==================================
# FRONTEND (Streamlit UI)
# ==================================
st.title("✨ Vibe Matcher")
st.markdown("Find fashion items that match your **mood or vibe** using AI-powered similarity search!")

query = st.text_input("🎯 Enter your vibe (e.g., 'urban chic', 'boho festival')")

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Button action
if st.button("Match My Vibe") and query.strip():
    with st.spinner("Finding your vibe..."):
        results, latency = get_top_matches(query)
    # Display top results
    st.subheader("Top Matches")
    for r in results:
        st.markdown(f"**{r['name']}** — {r['desc']}")
        st.caption(f"Vibes: {', '.join(r['vibes'])}")
        st.caption(f"Similarity: {r['score']:.3f}")
        st.divider()

    # Store result in session history
    st.session_state.history.append({
        "query": query,
        "top_match": results[0]["name"],
        "score": results[0]["score"],
        "latency": latency
    })

# ==================================
# SIDEBAR - History + Graphs
# ==================================
st.sidebar.header("🕒 Search History")
if len(st.session_state.history) > 0:
    # Display last 5 searches
    for h in reversed(st.session_state.history[-5:]):
        st.sidebar.write(f"**{h['query']}** → {h['top_match']}")
        st.sidebar.caption(f"⏱️ {h['latency']:.1f} ms | 🎯 {h['score']:.3f}")

    # ------------------------------------------
    # 📊 PERFORMANCE CHARTS (Latency + Score)
    # ------------------------------------------
    st.sidebar.subheader("📈 Performance Metrics")
    queries = [h["query"] for h in st.session_state.history]
    latencies = [h["latency"] for h in st.session_state.history]
    scores = [h["score"] for h in st.session_state.history]

    # Latency Chart (ms)
    fig1, ax1 = plt.subplots(figsize=(4, 2.4))
    ax1.barh(queries, latencies, color="#66b3ff", edgecolor="white")
    ax1.set_xlabel("Latency (ms)", fontsize=9, labelpad=5)
    ax1.set_ylabel("Query", fontsize=9)
    ax1.set_title("Query Processing Time", fontsize=11, pad=6)
    ax1.invert_yaxis()
    for i, v in enumerate(latencies):
        ax1.text(v + 10, i, f"{v:.0f} ms", va='center', fontsize=8, color='white')
    st.sidebar.pyplot(fig1, use_container_width=True)

    # Similarity Score Chart
    fig2, ax2 = plt.subplots(figsize=(4, 2.4))
    ax2.bar(queries, scores, color="#ffb366", edgecolor="white")
    ax2.set_ylabel("Similarity Score", fontsize=9)
    ax2.set_xlabel("Query", fontsize=9)
    ax2.set_title("Top Match Similarity", fontsize=11, pad=6)
    for i, v in enumerate(scores):
        ax2.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=8, color='white')
    plt.xticks(rotation=25, ha="right")
    st.sidebar.pyplot(fig2, use_container_width=True)
else:
    st.sidebar.info("No searches yet — run a query to view performance charts.")

# ==================================
# FOOTER
# ==================================
st.markdown("---")
st.caption("• Built by Hariom •")
