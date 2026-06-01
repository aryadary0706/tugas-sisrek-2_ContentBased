import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import re

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix RecSys – User-Based",
    page_icon="🎬",
    layout="wide",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ── */
body { font-family: 'Segoe UI', sans-serif; }

/* ── top banner ── */
.hero {
    background: linear-gradient(135deg, #141414 0%, #e50914 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    color: white;
}
.hero h1 { font-size: 2.4rem; margin: 0 0 6px; letter-spacing: -0.5px; }
.hero p  { font-size: 1.05rem; margin: 0; opacity: .85; }

/* ── section headings ── */
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e50914;
    border-left: 4px solid #e50914;
    padding-left: 10px;
    margin: 22px 0 12px;
}

/* ── user card ── */
.user-card {
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    color: #f1f1f1;
}
.user-card b { color: #e50914; font-size: 1rem; }

/* ── rec card ── */
.rec-card {
    background: #1a1a2e;
    border-left: 5px solid #e50914;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 14px;
    color: #f0f0f0;
}
.rec-card .rec-title { font-size: 1.05rem; font-weight: 700; color: #fff; }
.rec-card .rec-genre { font-size: 0.78rem; color: #aaa; margin: 4px 0; }
.rec-card .rec-desc  { font-size: 0.82rem; color: #ccc; line-height: 1.5; }
.score-badge {
    display: inline-block;
    background: #e50914;
    color: white;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 8px;
}

/* ── history pill ── */
.history-pill {
    display: inline-block;
    background: #2d2d2d;
    color: #f1f1f1;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    margin: 3px 4px 3px 0;
}

/* ── metric box ── */
.metric-box {
    background: #111;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
}
.metric-box .val { font-size: 2rem; font-weight: 800; color: #e50914; }
.metric-box .lbl { font-size: 0.85rem; color: #aaa; margin-top: 4px; }

/* ── info box ── */
.info-box {
    background: #0d0d0d;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 20px 24px;
    color: #ccc;
    line-height: 1.7;
    font-size: 0.9rem;
}
.info-box h4 { color: #e50914; margin-top: 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎬 Netflix Recommender System</h1>
  <p>Content-Based Filtering · SBERT Embeddings (all-MiniLM-L6-v2) · Cosine Similarity</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SYNTHETIC USER DATA  (10–15 users)
# ─────────────────────────────────────────────
USERS = {
    "U001": {
        "name": "Arya Kusuma",
        "age": 24, "gender": "Male", "city": "Jakarta",
        "preferred_genres": ["Thrillers", "Sci-Fi & Fantasy"],
        "watch_history": [
            {"title": "Bird Box", "rating": 5},
            {"title": "The Platform", "rating": 5},
            {"title": "Black Mirror: Bandersnatch", "rating": 4},
            {"title": "Inception", "rating": 5},
            {"title": "Annihilation", "rating": 3},
        ]
    },
    "U002": {
        "name": "Siti Rahayu",
        "age": 31, "gender": "Female", "city": "Bandung",
        "preferred_genres": ["Romantic Movies", "Comedies"],
        "watch_history": [
            {"title": "To All the Boys I've Loved Before", "rating": 5},
            {"title": "Always Be My Maybe", "rating": 4},
            {"title": "The Kissing Booth", "rating": 3},
            {"title": "Set It Up", "rating": 5},
        ]
    },
    "U003": {
        "name": "Budi Santoso",
        "age": 42, "gender": "Male", "city": "Surabaya",
        "preferred_genres": ["Documentaries", "Crime TV Shows"],
        "watch_history": [
            {"title": "Making a Murderer", "rating": 5},
            {"title": "The Keepers", "rating": 5},
            {"title": "Icarus", "rating": 4},
            {"title": "Wild Wild Country", "rating": 5},
            {"title": "Dirty Money", "rating": 4},
        ]
    },
    "U004": {
        "name": "Dewi Lestari",
        "age": 27, "gender": "Female", "city": "Yogyakarta",
        "preferred_genres": ["International Movies", "Dramas"],
        "watch_history": [
            {"title": "Parasite", "rating": 5},
            {"title": "Okja", "rating": 4},
            {"title": "The Wailing", "rating": 3},
            {"title": "Peninsula", "rating": 4},
        ]
    },
    "U005": {
        "name": "Reza Firmansyah",
        "age": 19, "gender": "Male", "city": "Medan",
        "preferred_genres": ["Action & Adventure", "Anime Features"],
        "watch_history": [
            {"title": "John Wick", "rating": 5},
            {"title": "Extraction", "rating": 4},
            {"title": "The Old Guard", "rating": 4},
            {"title": "6 Underground", "rating": 3},
        ]
    },
    "U006": {
        "name": "Nia Permata",
        "age": 35, "gender": "Female", "city": "Semarang",
        "preferred_genres": ["Horror Movies", "Thrillers"],
        "watch_history": [
            {"title": "The Haunting of Hill House", "rating": 5},
            {"title": "Haunted", "rating": 4},
            {"title": "The Ritual", "rating": 5},
            {"title": "His House", "rating": 4},
        ]
    },
    "U007": {
        "name": "Farhan Nugroho",
        "age": 29, "gender": "Male", "city": "Makassar",
        "preferred_genres": ["Stand-Up Comedy", "Comedies"],
        "watch_history": [
            {"title": "Dave Chappelle: Sticks & Stones", "rating": 5},
            {"title": "Ali Wong: Baby Cobra", "rating": 5},
            {"title": "John Mulaney: Kid Gorgeous at Radio City", "rating": 4},
            {"title": "Hannah Gadsby: Nanette", "rating": 5},
        ]
    },
    "U008": {
        "name": "Maya Indah",
        "age": 22, "gender": "Female", "city": "Denpasar",
        "preferred_genres": ["Children & Family Movies", "Animated"],
        "watch_history": [
            {"title": "Klaus", "rating": 5},
            {"title": "Over the Moon", "rating": 4},
            {"title": "The Willoughbys", "rating": 4},
            {"title": "Back to the Outback", "rating": 3},
        ]
    },
    "U009": {
        "name": "Hendra Wijaya",
        "age": 48, "gender": "Male", "city": "Palembang",
        "preferred_genres": ["Documentaries", "Science & Nature TV"],
        "watch_history": [
            {"title": "Our Planet", "rating": 5},
            {"title": "Night on Earth", "rating": 4},
            {"title": "72 Dangerous Animals: Asia", "rating": 4},
            {"title": "Abstract: The Art of Design", "rating": 5},
        ]
    },
    "U010": {
        "name": "Ratna Sari",
        "age": 33, "gender": "Female", "city": "Balikpapan",
        "preferred_genres": ["TV Dramas", "Korean TV Shows"],
        "watch_history": [
            {"title": "Crash Landing on You", "rating": 5},
            {"title": "Itaewon Class", "rating": 5},
            {"title": "Extracurricular", "rating": 4},
            {"title": "My Holo Love", "rating": 3},
        ]
    },
    "U011": {
        "name": "Dimas Prasetyo",
        "age": 25, "gender": "Male", "city": "Tangerang",
        "preferred_genres": ["Sci-Fi & Fantasy", "Action & Adventure"],
        "watch_history": [
            {"title": "Stranger Things", "rating": 5},
            {"title": "Dark", "rating": 5},
            {"title": "The OA", "rating": 4},
            {"title": "Altered Carbon", "rating": 4},
        ]
    },
    "U012": {
        "name": "Laila Azzahra",
        "age": 30, "gender": "Female", "city": "Padang",
        "preferred_genres": ["Crime TV Shows", "Thrillers"],
        "watch_history": [
            {"title": "Mindhunter", "rating": 5},
            {"title": "Narcos", "rating": 5},
            {"title": "Ozark", "rating": 5},
            {"title": "Bloodline", "rating": 3},
        ]
    },
    "U013": {
        "name": "Wahyu Andika",
        "age": 38, "gender": "Male", "city": "Pekanbaru",
        "preferred_genres": ["Sports Movies", "Documentaries"],
        "watch_history": [
            {"title": "The Last Dance", "rating": 5},
            {"title": "Formula 1: Drive to Survive", "rating": 5},
            {"title": "Sunderland 'Til I Die", "rating": 4},
            {"title": "Losers", "rating": 4},
        ]
    },
}

# ─────────────────────────────────────────────
#  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="📥 Memuat dataset Netflix...")
def load_data():
    try:
        df = pd.read_csv("netflix_titles.csv")
    except FileNotFoundError:
        st.error("❌ File `netflix_titles.csv` tidak ditemukan. Letakkan file CSV di direktori yang sama dengan `app.py`.")
        st.stop()

    features = ['title', 'director', 'cast', 'country', 'listed_in', 'description']
    for f in features:
        df[f] = df[f].fillna('')

    def clean_names(text):
        if isinstance(text, str):
            text = text.lower()
            names = text.split(',')
            return " ".join([n.replace(" ", "") for n in names])
        return ""

    df['cast_cleaned'] = df['cast'].apply(clean_names)
    df['director_cleaned'] = df['director'].apply(clean_names)

    def combine_features(row):
        return (str(row['title']) + " " +
                row['director_cleaned'] + " " +
                row['cast_cleaned'] + " " +
                str(row['country']) + " " +
                str(row['listed_in']) + " " +
                str(row['description']))

    df['combined_features'] = df.apply(combine_features, axis=1).str.lower()
    return df


@st.cache_resource(show_spinner="🤖 Membuat SBERT embeddings (tunggu sebentar)...")
def build_embeddings(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
    return embeddings


def get_recommendations(user_history, df, embeddings, top_n=10):
    indices = []
    for item in user_history:
        match = df[df['title'].str.lower() == item['title'].lower()]
        if not match.empty:
            indices.append(match.index[0])

    if not indices:
        return pd.DataFrame()

    user_vecs = embeddings[indices]
    profile_vec = np.mean(user_vecs, axis=0).reshape(1, -1)
    scores = cosine_similarity(profile_vec, embeddings)[0]

    df_scores = df.copy()
    df_scores['similarity_score'] = scores
    df_filtered = df_scores.drop(index=indices)
    return df_filtered.sort_values('similarity_score', ascending=False).head(top_n)


def evaluate_user(user_history, df, embeddings, k=5):
    results = {"precision": [], "recall": [], "f1": []}
    for item in user_history:
        match = df[df['title'].str.lower() == item['title'].lower()]
        if match.empty:
            continue
        idx = match.index[0]
        target_genres = df.loc[idx, 'listed_in']
        scores = cosine_similarity(embeddings[idx].reshape(1, -1), embeddings)[0]
        top_idx = np.argsort(scores)[::-1][1:k+1]
        recs = df.iloc[top_idx]

        def relevance(rec_genres):
            t = set(g.strip().lower() for g in target_genres.split(','))
            r = set(g.strip().lower() for g in rec_genres.split(','))
            return 1 if t & r else 0

        y_true = [relevance(row['listed_in']) for _, row in recs.iterrows()]
        y_pred = [1] * k
        results["precision"].append(precision_score(y_true, y_pred, zero_division=0))
        results["recall"].append(recall_score(y_true, y_pred, zero_division=0))
        results["f1"].append(f1_score(y_true, y_pred, zero_division=0))

    if not results["precision"]:
        return None
    return {k: round(float(np.mean(v)), 3) for k, v in results.items()}


# ─────────────────────────────────────────────
#  SIDEBAR – User Selection & Info
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Pilih Active User")

    user_labels = {uid: f"{uid} – {info['name']}" for uid, info in USERS.items()}
    selected_uid = st.selectbox(
        "User ID",
        options=list(USERS.keys()),
        format_func=lambda uid: user_labels[uid]
    )

    top_n = st.slider("Jumlah rekomendasi", min_value=3, max_value=20, value=8)

    st.markdown("---")
    st.markdown("**Daftar semua User ID:**")
    for uid, info in USERS.items():
        marker = "🟥" if uid == selected_uid else "⚪"
        st.markdown(f"{marker} `{uid}` {info['name']}")


# ─────────────────────────────────────────────
#  LOAD DATA  &  EMBEDDINGS
# ─────────────────────────────────────────────
df = load_data()
embeddings = build_embeddings(df['combined_features'].tolist())
user = USERS[selected_uid]

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Rekomendasi", "📊 Evaluasi & Metrik", "ℹ️ Tentang Sistem"])

# ────────── TAB 1 : REKOMENDASI ──────────
with tab1:
    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Profil User</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="user-card">
          <b>{user['name']}</b><br>
          🆔 {selected_uid} &nbsp;|&nbsp; 🎂 {user['age']} thn &nbsp;|&nbsp; {user['gender']}<br>
          🌆 {user['city']}<br><br>
          <span style="color:#aaa; font-size:.82rem;">Genre Favorit:</span><br>
          {"".join(f'<span class="history-pill">🏷️ {g}</span>' for g in user["preferred_genres"])}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Riwayat Tontonan</div>', unsafe_allow_html=True)
        for movie in user['watch_history']:
            stars = "⭐" * movie['rating']
            st.markdown(f"""
            <div class="user-card" style="padding:10px 16px;">
              🎥 <b>{movie['title']}</b><br>
              <span style="color:#f5c518;">{stars}</span>
              <span style="color:#888; font-size:.78rem;"> ({movie['rating']}/5)</span>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-title">🔎 Rekomendasi Konten</div>', unsafe_allow_html=True)

        with st.spinner("Menghitung profil user dan mencari rekomendasi..."):
            recs = get_recommendations(user['watch_history'], df, embeddings, top_n=top_n)

        if recs.empty:
            st.warning("Tidak ada film dari riwayat user yang cocok dengan katalog. Pastikan judul film ada di netflix_titles.csv.")
        else:
            for i, (_, row) in enumerate(recs.iterrows(), 1):
                score_pct = f"{row['similarity_score']*100:.1f}%"
                st.markdown(f"""
                <div class="rec-card">
                  <div style="float:right"><span class="score-badge">🎯 {score_pct}</span></div>
                  <div class="rec-title">#{i} &nbsp;{row['title']}</div>
                  <div class="rec-genre">🏷️ {row['listed_in']}</div>
                  <div class="rec-desc">{row['description'][:220]}{"..." if len(row['description'])>220 else ""}</div>
                </div>
                """, unsafe_allow_html=True)


# ────────── TAB 2 : EVALUASI ──────────
with tab2:
    st.markdown('<div class="section-title">📐 Metrik Evaluasi untuk User Aktif</div>', unsafe_allow_html=True)
    st.info("Evaluasi menggunakan **genre overlap** sebagai *ground truth*: rekomendasi dianggap relevan jika memiliki setidaknya satu genre yang sama dengan film di riwayat user.")

    with st.spinner("Menghitung Precision, Recall, F1-Score..."):
        metrics = evaluate_user(user['watch_history'], df, embeddings, k=5)

    if metrics:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-box">
              <div class="val">{metrics['precision']:.2f}</div>
              <div class="lbl">Mean Precision@5</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-box">
              <div class="val">{metrics['recall']:.2f}</div>
              <div class="lbl">Mean Recall@5</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-box">
              <div class="val">{metrics['f1']:.2f}</div>
              <div class="lbl">Mean F1-Score@5</div></div>""", unsafe_allow_html=True)
    else:
        st.warning("Tidak dapat menghitung metrik – judul film di riwayat user tidak ditemukan di katalog.")

    st.markdown("---")
    st.markdown('<div class="section-title">📖 Penjelasan Metrik</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="info-box">
          <h4>🎯 Precision@K</h4>
          Dari <b>K</b> item yang direkomendasikan, berapa persentase yang benar-benar relevan?<br><br>
          <code>Precision@K = Relevan ∩ Direkomendasikan / K</code><br><br>
          Semakin tinggi = semakin sedikit rekomendasi "tidak berguna" yang diberikan ke user.
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="info-box">
          <h4>📡 Recall@K</h4>
          Dari semua item relevan yang ada, berapa yang berhasil <b>ditemukan</b> oleh sistem dalam Top-K?<br><br>
          <code>Recall@K = Relevan ∩ Direkomendasikan / Total Relevan</code><br><br>
          Nilai Recall = Precision pada Top-K karena total relevan tidak dibatasi.
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="info-box">
          <h4>⚖️ F1-Score@K</h4>
          Rata-rata harmonik antara Precision dan Recall.<br><br>
          <code>F1 = 2 × (P × R) / (P + R)</code><br><br>
          Memberikan skor tunggal yang menyeimbangkan keduanya. Berguna sebagai metrik utama evaluasi.
        </div>
        """, unsafe_allow_html=True)


# ────────── TAB 3 : ABOUT ──────────
with tab3:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="section-title">📦 Dataset</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
          <h4>Netflix Movies and TV Shows</h4>
          Dataset berisi metadata konten yang tersedia di platform Netflix, mencakup:<br><br>
          <b>Total konten:</b> {len(df):,} judul<br>
          <b>Movies:</b> {len(df[df['type']=='Movie']):,} judul<br>
          <b>TV Shows:</b> {len(df[df['type']=='TV Show']):,} judul<br><br>
          <b>Fitur utama yang digunakan:</b><br>
          • <code>title</code> – Judul konten<br>
          • <code>director</code> – Nama sutradara<br>
          • <code>cast</code> – Daftar pemain<br>
          • <code>country</code> – Negara produksi<br>
          • <code>listed_in</code> – Kategori/genre<br>
          • <code>description</code> – Sinopsis singkat<br><br>
          Semua fitur teks digabung menjadi satu <b>metadata soup</b> sebelum di-encode.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">🔧 Preprocessing</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          <h4>Pipeline Preprocessing</h4>
          1. <b>Imputasi NaN</b> – Nilai kosong diisi string kosong <code>''</code><br>
          2. <b>Normalisasi nama</b> – Nama aktor/sutradara: lowercase + hapus spasi antar kata (mis. <i>"kirsten johnson" → "kirstenjohnson"</i>) agar model tidak memisahkan sebagai token berbeda<br>
          3. <b>Metadata Soup</b> – Semua kolom digabungkan menjadi satu string teks panjang<br>
          4. <b>Lowercase</b> – Seluruh teks dikonversi ke huruf kecil untuk konsistensi embedding
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">🤖 Metode Embedding (SBERT)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          <h4>Sentence-BERT: all-MiniLM-L6-v2</h4>
          Model ini dipilih karena beberapa alasan kuat:<br><br>

          <b>✅ Semantic Understanding</b><br>
          Berbeda dengan TF-IDF yang hanya menghitung frekuensi kata, SBERT memahami <i>makna semantik</i> dari teks. Dua deskripsi berbeda kata namun bermakna sama akan menghasilkan vektor yang berdekatan.<br><br>

          <b>✅ Efisiensi Tinggi</b><br>
          Model <code>all-MiniLM-L6-v2</code> adalah versi distilasi dari model besar. Ukurannya sangat kecil (~22MB) namun akurasinya tetap kompetitif untuk tugas semantic similarity.<br><br>

          <b>✅ Vektor Berdimensi Tetap</b><br>
          Menghasilkan vektor 384 dimensi per teks, berapapun panjang inputnya. Ini memudahkan komputasi cosine similarity secara matriks.<br><br>

          <b>✅ Pretrained pada Data Besar</b><br>
          Dilatih pada jutaan pasangan kalimat dari berbagai sumber (Wikipedia, Reddit, dll), sehingga punya pemahaman bahasa yang kaya tanpa perlu fine-tuning.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">🔄 Alur Sistem Rekomendasi</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          <h4>User-Based Content Filtering</h4>
          1. <b>Ambil riwayat tontonan</b> user aktif (judul + rating)<br>
          2. <b>Cari embedding</b> setiap film di riwayat dari matriks embedding global<br>
          3. <b>Agregasi profil</b> – rata-rata (<i>mean</i>) seluruh vektor film menjadi satu <b>User Profile Vector</b><br>
          4. <b>Cosine Similarity</b> – hitung kemiripan profil user dengan semua film di katalog<br>
          5. <b>Filter & Ranking</b> – hilangkan film yang sudah ditonton, urutkan skor tertinggi<br>
          6. <b>Tampilkan Top-N</b> rekomendasi beserta skor kemiripan
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:.8rem;'>"
    "🎬 Netflix Recommender System &nbsp;|&nbsp; Tugas Kelompok 2 &nbsp;|&nbsp; "
    "SBERT · Cosine Similarity · Content-Based Filtering"
    "</div>",
    unsafe_allow_html=True
)