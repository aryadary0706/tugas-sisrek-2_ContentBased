import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF warnings

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

#  PAGE CONFIG
st.set_page_config(
    page_title="Tugas Besar Recsys: Content-Based",
    page_icon="🎬",
    layout="wide",
)

#  CUSTOM CSS 
st.markdown("""
<style>
/*  global  */
body { font-family: 'Segoe UI', sans-serif; }

/*  top banner  */
.hero {
    background: #d13e3e;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    color: white;
}
.hero h1 { font-size: 2.4rem; margin: 0 0 6px; letter-spacing: -0.5px; }
.hero p  { font-size: 1.05rem; margin: 0; opacity: .85; }

/*  section headings  */
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e50914;
    border-left: 4px solid #e50914;
    padding-left: 10px;
    margin: 22px 0 12px;
}

/*  user card  */
.user-card {
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    color: #f1f1f1;
}
.user-card b { color: #e50914; font-size: 1rem; }

/*  rec card  */
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

/*  history pill  */
.history-pill {
    display: inline-block;
    background: #2d2d2d;
    color: #f1f1f1;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.8rem;
    margin: 3px 4px 3px 0;
}

/*  metric box  */
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

/*  info box  */
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

#  HERO BANNER
st.markdown("""
<div class="hero">
  <h1>Tugas Kelompok 2</h1>
  <p>Content-Based Filtering menggunakan SBERT Embeddings (all-MiniLM-L6-v2) · Evaluasi: Hit Rate · Precision@K · F1 · DCG · NDCG</p>
</div>
""", unsafe_allow_html=True)


USERS = {
    "U001": {
        "name": "Arya Kusuma",
        "age": 24,
        "preferred_genres": ["Thrillers", "Sci-Fi & Fantasy", "Action & Adventures"],
        "watch_history": [
            {"title": "Bird Box"},
            {"title": "The Platform"},
            {"title": "Black Mirror: Bandersnatch"},
            {"title": "Inception"},
            {"title": "Doom: Annihilation"},
            {"title": "Intrusion"},
            {"title": "Squid Game"},
        ]
    },
    "U002": {
        "name": "Siti Rahayu",
        "age": 31,
        "preferred_genres": ["Romantic Movies", "Comedies"],
        "watch_history": [
            {"title": "Valentine's Day"},
            {"title": "Twilight"},
            {"title": "Home Again"},
        ]
    },
    "U003": {
        "name": "Budi Santoso",
        "age": 28,
        "preferred_genres": ["Dramas", "International Movies"],
        "watch_history": [
            {"title": "Gie"},
            {"title": "Home Again"},
            {"title": "Air Force One"},
            {"title": "Charlie's Angels"},
        ]
    },
    "U004": {
        "name": "Dewi Lestari",
        "age": 27,
        "preferred_genres": ["Horror Movies", "Thrillers", "Drama"],
        "watch_history": [
        ]
    },
    "U005": {
        "name": "Reza Firmansyah",
        "age": 19,
        "preferred_genres": ["Action & Adventure", "TV Horror"],
        "watch_history": [
            {"title": "Eerie"},
            {"title": "Extraction"},
            {"title": "The Old Guard"},
            {"title": "6 Underground"},
        ]
    },
    "U006": {
        "name": "Nia Permata",
        "age": 35,
        "preferred_genres": ["Documentaries", "Crime"],
        "watch_history": [
            {"title": "Our Planet"},
            {"title": "Making a Murderer"},
        ]
    },
    "U007": {
        "name": "Farhan Nugroho",
        "age": 29,
        "preferred_genres": ["Comedies", "Science & Nature"], 
        "watch_history": [
            {"title": "Age of Tanks"},
            {"title": "Arjun: The Warrior Prince"},
            {"title": "Frontier"},
            {"title": "Sick Note"},
            {"title": "Target"},
        ]
    },
    "U008": {
        "name": "Maya Indah",
        "age": 22,
        "preferred_genres": ["Children & Family Movies"], 
        "watch_history": [
            {"title": "Westside"},
            {"title": "Damnation"},
            {"title": "Follow This"},
            {"title": "Gun City"},
            {"title": "Gnome Alone"},
        ]
    },
    "U009": {
        "name": "Hendra Wijaya",
        "age": 45,
        "preferred_genres": [],
        "watch_history": [
            {"title": "Inception"},
            {"title": "The Matrix"},
            {"title": "Jailbirds New Orleans"},
            {"title": "Sankofa"},
            {"title": "A Silent Voice"},
        ]
    },
    "U010": {
        "name": "Ratna Sari",
        "age": 33,
        "preferred_genres": [],
        "watch_history": [
        ]
    },
}

#  DATA LOADING & PREPROCESSING
def clean_names(text):
    if isinstance(text, str):
        text = text.lower()
        names = text.split(',')
        return " ".join([n.replace(" ", "") for n in names])
    return ""
    
def clean_genres(genre_string):
    # Buang Stop Words, clean simbol ('&', ','), penghilangan spasi
    if pd.isna(genre_string) or not isinstance(genre_string, str):
        return []
    # Ganti '&' dan ',' dengan spasi, ubah ke lowercase
    cleaned = genre_string.lower().replace('&', ' ').replace(',', ' ')
    words = cleaned.split()
    # Buang kata-kata sampah yang berulang
    stop_words = {'tv', 'shows', 'movies', 'show', 'movie'}
    filtered_words = [w for w in words if w not in stop_words]
    return filtered_words

@st.cache_data(show_spinner="📥 Memuat dataset Netflix...")
def load_data():
    try:
        df = pd.read_csv("netflix_titles.csv")
    except FileNotFoundError:
        st.error("❌ File `netflix_titles.csv` tidak ditemukan. Letakkan file CSV di direktori yang sama dengan `app.py`.")
        st.stop()

    features = ['title', 'rating', 'listed_in', 'description', 'director', 'cast']
    for f in features:
        df[f] = df[f].fillna('')
    
    df['cast_cleaned'] = df['cast'].apply(clean_names)
    df['director_cleaned'] = df['director'].apply(clean_names)
    df['listed_in_cleaned'] = df['listed_in'].apply(clean_genres)
    def combine_features(row):
        return " ".join([str(row['title']), str(row['rating']), " ".join(row['listed_in_cleaned']), str(row['description']), str(row['director_cleaned']), str(row['cast_cleaned'])])

    df['combined_features'] = df.apply(combine_features, axis=1).str.lower()
    return df

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner="🤖 Membuat SBERT embeddings (tunggu sebentar)...")
def build_embeddings(texts, _model):
    embeddings = _model.encode(texts, show_progress_bar=False, batch_size=64)
    return embeddings

# 2. Fungsi Cek Semua Genre Unik di Dataset
def get_all_unique_genres(dataframe):
    unique_genres = set()
    for listed_in in dataframe['listed_in'].dropna():
        unique_genres.update(clean_genres(listed_in))
    return sorted(list(unique_genres))

def get_max_allowed_rating(age):
    if age < 7: return ['G', 'TV-Y', 'TV-G']
    elif age < 13: return ['G', 'TV-Y', 'TV-G', 'PG', 'TV-Y7', 'TV-Y7-FV', 'TV-PG']
    elif age < 14: return ['G', 'TV-Y', 'TV-G', 'PG', 'TV-Y7', 'TV-Y7-FV', 'TV-PG', 'PG-13']
    elif age < 17: return ['G', 'TV-Y', 'TV-G', 'PG', 'TV-Y7', 'TV-Y7-FV', 'TV-PG', 'PG-13', 'TV-14']
    else: return ['G', 'TV-Y', 'TV-G', 'PG', 'TV-Y7', 'TV-Y7-FV', 'TV-PG', 'PG-13', 'TV-14', 'R', 'NC-17', 'TV-MA', 'NR', 'UR']


# 4. Fungsi Utama Rekomendasi (Mengatasi Cold-Start & Fallback)
def get_recommendations(user_data, df_catalog, embeddings_matrix):
    age = user_data.get('age', 18)
    watch_history = user_data.get('watch_history', [])
    preferred_genres = user_data.get('preferred_genres', [])
    
    # Ambil rating[] yang ada di user
    allowed_ratings = get_max_allowed_rating(age)

    df_scores = df_catalog.copy()
    df_scores['rating'] = df_scores['rating'].fillna('NR')
    df_scores = df_scores[df_scores['rating'].isin(allowed_ratings)].reset_index(drop=True)
    
    valid_indices = df_catalog[df_catalog['rating'].fillna('NR').isin(allowed_ratings)].index

    filtered_embeddings = embeddings_matrix[valid_indices]
    
    catalog_genres = get_all_unique_genres(df_catalog)
    clean_user_pref = [w for genre in preferred_genres for w in clean_genres(genre) if w in catalog_genres]

    history_titles = [m['title'].lower() for m in watch_history]

    history_indices_in_filtered = df_scores[df_scores['title'].str.lower().isin(history_titles)].index.tolist()
    
    scores = np.zeros(len(df_scores))
    if not watch_history or not history_indices_in_filtered:
        if clean_user_pref:
            # ambil genre disukai oleh user jika watch_history kosong
            pref_text = " ".join(clean_user_pref)
            user_profile_vector = model.encode([pref_text]).reshape(1, -1)
            scores = cosine_similarity(user_profile_vector, filtered_embeddings)[0]
        else:
            # Ambil list item dengan release_year terdekat
            df_scores['release_year'] = pd.to_numeric(df_scores['release_year'], errors='coerce')
            df_fallback = df_scores.sort_values('release_year', ascending=False).head(5)
            df_fallback['similarity_score'] = 0.0
            return df_fallback
    else:
        user_vecs = filtered_embeddings[history_indices_in_filtered] 
        user_profile_vector = np.mean(user_vecs, axis=0).reshape(1, -1)
        scores = cosine_similarity(user_profile_vector, filtered_embeddings)[0]

    df_scores['similarity_score'] = scores
    # DataFrame hasil akhir penyaringan di mana seluruh film yang judulnya terdaftar di dalam history_titles sudah dieliminasi agar user tidak direkomendasikan film yang sudah pernah ia tonton sebelumnya.
    df_final = df_scores[~df_scores['title'].str.lower().isin(history_titles)]
    return df_final.sort_values('similarity_score', ascending=False).head(5)

# 5. Fungsi Evaluasi Metrik
def evaluate_recommendations(recommendations, user_data, df_catalog):
    if recommendations.empty:
        return { "hit_rate": 0, "f1_score": 0, "precision_at_k": 0, "dcg": 0.0, "ndcg": 0.0, "relevance_scores": [], "ideal_relevance_scores": [] }

    #  Bangun kumpulan kata target dari preferred genres + watch history 
    user_target_words = set()
    for g in user_data.get('preferred_genres', []):
        user_target_words.update(clean_genres(g))

    for item in user_data.get('watch_history', []):
        match = df_catalog[df_catalog['title'].str.lower() == item['title'].lower()]
        if not match.empty:
            row = match.iloc[0]
            user_target_words.update(clean_genres(row['listed_in']))
            if row['director']:
                user_target_words.update(row['director'].lower().replace(',', ' ').split())
            if row['cast']:
                user_target_words.update(row['cast'].lower().replace(',', ' ').split())

    hits = 0
    f1_scores = []
    relevance_scores = []   # Graded relevance per posisi (0-3 scale)

    for _, row in recommendations.iterrows():
        rec_words = set(clean_genres(row['listed_in']))
        if row['director']:
            rec_words.update(row['director'].lower().replace(',', ' ').split())
        if row['cast']:
            rec_words.update(row['cast'].lower().replace(',', ' ').split())

        intersection = user_target_words.intersection(rec_words)

        #  Hit 
        if len(intersection) > 0:
            hits += 1

        #  F1 
        if len(rec_words) == 0 or len(user_target_words) == 0:
            f1 = 0.0
        else:
            precision = len(intersection) / len(rec_words)
            recall    = len(intersection) / len(user_target_words)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

        #  Graded Relevance (0-3) berdasarkan overlap ratio 
        if len(user_target_words) == 0:
            rel = 0
        else:
            overlap_ratio = len(intersection) / len(user_target_words)
            if overlap_ratio >= 0.15:
                rel = 3
            elif overlap_ratio >= 0.08:
                rel = 2
            elif overlap_ratio > 0:
                rel = 1
            else:
                rel = 0
        relevance_scores.append(rel)

    k = len(relevance_scores)

    #  Precision@K 
    precision_at_k = hits / k if k > 0 else 0.0

    #  DCG  (formula 9 dari slide: rel_i / log2(i+1)) 
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))

    #  IDCG – ideal ordering (sort descending) 
    ideal_rels = sorted(relevance_scores, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

    #  NDCG 
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    hit_rate = 1 if hits > 0 else 0

    return {
        "hit_rate":              hit_rate,
        "f1_score":              round(float(np.mean(f1_scores)), 4),
        "precision_at_k":        round(float(precision_at_k), 4),
        "dcg":                   round(float(dcg), 4),
        "ndcg":                  round(float(ndcg), 4),
        "relevance_scores":      relevance_scores,
        "ideal_relevance_scores": ideal_rels,
    }

#  SIDEBAR – User Selection & Info
with st.sidebar:
    st.markdown("## 👤 Pilih Active User")
    
    if "selected_uid" not in st.session_state:
        st.session_state.selected_uid = "U001"

    st.markdown("Klik pada nama user di bawah untuk mengganti profil aktif:")
    st.markdown("---")
    
    for uid, info in USERS.items():
        is_active = (uid == st.session_state.selected_uid)
        marker = "🟥" if is_active else "⚪"
        
        button_label = f"{marker} {uid} — {info['name']}"
        
        btn_type = "primary" if is_active else "secondary"
        
        if st.button(button_label, key=f"btn_{uid}", use_container_width=True, type=btn_type):
            st.session_state.selected_uid = uid
            st.rerun() 

    selected_uid = st.session_state.selected_uid

#  LOAD DATA  &  EMBEDDINGS
# Cache data dan embeddings
df = load_data()
model = load_model()

# Ambil data user yang dipilih
user = USERS[selected_uid]

# Bangun matriks embedding untuk seluruh katalog (sekali saja, cached)
embeddings = build_embeddings(df['combined_features'].tolist(), model)

#  FUNGSI REKOMENDASI & EVALUASI
reccommendations_df = get_recommendations(user, df, embeddings)
metrics = evaluate_recommendations(reccommendations_df, user, df)

#  TABS
tab1, tab2, tab3 = st.tabs(["🎯 Rekomendasi", "📊 Evaluasi & Metrik", "ℹ️ Tentang Sistem"])

#  TAB 1 : REKOMENDASI 
with tab1:
    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Profil User</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="user-card">
          <b>{user['name']}</b><br>
          🆔 {selected_uid} &nbsp;|&nbsp; 🎂 {user['age']} thn &nbsp;|&nbsp;
          <span style="color:#aaa; font-size:.82rem;">Genre Favorit:</span><br>
          {"".join(f'<span class="history-pill">🏷️ {g}</span>' for g in user["preferred_genres"])}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Riwayat Tontonan</div>', unsafe_allow_html=True)
        for movie in user['watch_history']:
            # Ambil rating asli dari catalog jika ada, atau fallback ke teks default
            match = df[df['title'].str.lower() == movie['title'].lower()]
            rating_content = match.iloc[0]['rating'] if not match.empty else "N/A"
            genres_content = match.iloc[0]['listed_in'] if not match.empty else "N/A"
            
            st.markdown(f"""
            <div class="user-card" style="padding:10px 16px;">
              🎥 <b>{movie['title']}</b><br>
              <span style="color:#888; font-size:.78rem;"> Rating: {rating_content}</span><br/>
              <span style="color:#888; font-size:.78rem;"> Genre: {genres_content}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-title">🔎 Rekomendasi Konten</div>', unsafe_allow_html=True)
        if reccommendations_df.empty:
            st.warning("Tidak ada film yang cocok dengan profil user atau batasan usia.")
        else:
            for i, (_, row) in enumerate(reccommendations_df.iterrows(), 1):
                score_pct = f"{row['similarity_score']*100:.1f}%"
                st.markdown(f"""
                <div class="rec-card">
                  <div style="float:right"><span class="score-badge">🎯 {score_pct}</span></div>
                  <div class="rec-title">#{i} &nbsp;{row['title']} <span style="font-size:0.8rem; color:#e50914;">({row['rating']})</span></div>
                  <div class="rec-genre">🏷️ {row['listed_in']}</div>
                  <div class="rec-desc">{row['description'][:220]}{"..." if len(row['description'])>220 else ""}</div>
                </div>
                """, unsafe_allow_html=True)


#  TAB 2 : EVALUASI 
with tab2:
    st.markdown('<div class="section-title">📐 Metrik Evaluasi untuk User Aktif</div>', unsafe_allow_html=True)
    st.info(
        "Evaluasi menggunakan **genre, director, dan cast overlap** sebagai *ground truth*. "
        "Relevansi dinilai dengan skala **0-3** (graded relevance), sesuai kerangka **DCG/NDCG** dari Top-N evaluation."
    )

    if metrics and not reccommendations_df.empty:
        #  Baris 1: 4 metrik utama 
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            hr_label = "✅ Hit" if metrics['hit_rate'] == 1 else "❌ Miss"
            st.markdown(f"""<div class="metric-box">
              <div class="val">{metrics['hit_rate']}</div>
              <div class="lbl">Hit Rate<br><span style="font-size:.75rem;color:#888;">{hr_label}</span></div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-box">
              <div class="val">{metrics['precision_at_k']:.4f}</div>
              <div class="lbl">Precision@K<br><span style="font-size:.75rem;color:#888;">K = {len(metrics['relevance_scores'])}</span></div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-box">
              <div class="val">{metrics['f1_score']:.4f}</div>
              <div class="lbl">Mean F1-Score<br><span style="font-size:.75rem;color:#888;">Overlap Fitur</span></div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-box">
              <div class="val">{metrics['ndcg']:.4f}</div>
              <div class="lbl">NDCG<br><span style="font-size:.75rem;color:#888;">Normalized DCG</span></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        #  Baris 2: DCG detail + tabel relevance 
        col_dcg, col_tbl = st.columns([1, 2], gap="large")

        with col_dcg:
            st.markdown(f"""<div class="metric-box" style="margin-top:4px;">
              <div class="val">{metrics['dcg']:.4f}</div>
              <div class="lbl">DCG (Discounted Cumulative Gain)<br>
              <span style="font-size:.75rem;color:#888;">IDCG = {sum(r / np.log2(i+2) for i,r in enumerate(metrics['ideal_relevance_scores'])):.4f}</span>
              </div>
            </div>""", unsafe_allow_html=True)

        with col_tbl:
            rel_data = []
            for i, (_, row) in enumerate(reccommendations_df.iterrows()):
                rel  = metrics['relevance_scores'][i]
                disc = round(rel / np.log2(i + 2), 4)
                stars = "⭐" * rel + "" * (3 - rel)
                rel_data.append({
                    "Pos": f"#{i+1}",
                    "Judul Film": row['title'][:35] + ("…" if len(row['title']) > 35 else ""),
                    "rel_i (0-3)": f"{rel}  {stars}",
                    "log₂(i+1)": round(np.log2(i + 2), 3),
                    "rel_i / log₂(i+1)": disc,
                })
            st.markdown("**Detail perhitungan DCG per posisi:**")
            st.dataframe(pd.DataFrame(rel_data), use_container_width=True, hide_index=True)

    else:
        st.warning("Tidak dapat menghitung metrik — Data rekomendasi kosong atau riwayat user tidak ditemukan di katalog.")

    st.markdown("---")
    st.markdown('<div class="section-title">📖 Penjelasan Metrik</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="info-box">
          <h4>🎯 Hit Rate</h4>
          Apakah sistem berhasil memberikan <b>minimal 1 item relevan</b> dalam Top-N rekomendasi?<br><br>
          <code>1 = Sukses (ada overlap fitur)</code><br>
          <code>0 = Gagal (tidak ada overlap sama sekali)</code>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          <h4>📊 DCG — Discounted Cumulative Gain</h4>
          Item relevan yang muncul di <b>posisi lebih atas</b> diberi bobot lebih tinggi.
          Relevansi di-diskon secara logaritmik sesuai posisinya:<br><br>
          <code>DCG = Σ rel_i / log₂(i+1)</code><br><br>
          Semakin tinggi nilai DCG, semakin baik urutan rekomendasi sistem.
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="info-box">
          <h4>🎯 Precision@K</h4>
          Proporsi item relevan dari seluruh K item yang direkomendasikan.<br><br>
          <code>Precision@K = Jumlah Hit / K</code><br><br>
          Berbeda dengan Hit Rate (biner), Precision@K mengukur <b>seberapa banyak</b> item relevan yang berhasil ditampilkan.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          <h4>🏆 NDCG — Normalized DCG</h4>
          DCG dinormalisasi terhadap nilai DCG ideal (IDCG), yaitu kondisi semua item relevan diurutkan sempurna dari atas.<br><br>
          <code>NDCG = DCG / IDCG</code><br><br>
          Nilai mendekati <b>1.0</b> berarti urutan rekomendasi sangat mendekati urutan ideal.
          Ini adalah metrik Top-N paling komprehensif sesuai materi kuliah.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      <h4>⚖️ F1-Score (Feature Overlap)</h4>
      Rata-rata harmonik antara <b>precision</b> dan <b>recall</b> dari kemunculan kata kunci target
      (genre, sutradara, aktor) di setiap film rekomendasi dibandingkan profil user.<br><br>
      F1-Score melengkapi NDCG karena mengukur kualitas konten secara keseluruhan, bukan hanya urutan ranking.
    </div>
    """, unsafe_allow_html=True)


#  TAB 3 : ABOUT 
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
            <code>title</code> - Judul konten<br>
            <code>rating</code> - Rating umur konten<br>
            <code>director</code> - Nama sutradara<br>
            <code>cast</code> - Daftar pemain<br>
            <code>listed_in</code> - Kategori/genre<br>
            <code>description</code> - Deskripsi Film/Series<br><br>
          Semua fitur teks digabung menjadi satu <b>metadata soup</b> sebelum di-encode.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">🔧 Preprocessing</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
          <h4>Pipeline Preprocessing</h4>
          1. <b>Imputasi NaN</b> - Nilai kosong diisi string kosong <code>''</code><br>
          2. <b>Normalisasi nama</b> - Nama aktor/sutradara: lowercase + hapus spasi antar kata (mis. <i>"kirsten johnson" → "kirstenjohnson"</i>) agar model tidak memisahkan sebagai token berbeda<br>
          3. <b>Metadata Soup</b> - Semua kolom digabungkan menjadi satu string teks panjang<br>
          4. <b>Lowercase</b> - Seluruh teks dikonversi ke huruf kecil untuk konsistensi embedding
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
            <h4>Content-Based Filtering</h4>
            1. <b>Embeddings Global</b> seluruh data pada dataset film (kombinasi fitur yang dipakai) <br>
            2. <b>Ambil riwayat tontonan</b> user aktif (judul, deskripsi, rating, cast, director)<br>
            3. <b>Penanganan Usia (rating)</b> memfilter konten berdasarkan batas usia pengguna sebelum menghitung similarity<br>
            4. <b>Cari embedding</b> setiap film di riwayat dari matriks embedding global<br>
            5. <b>Agregasi profil</b> = rata-rata (<i>mean</i>) seluruh vektor film menjadi satu <b>User Profile Vector Jika ada history tontonan</b><br>
            6. <b>Penanganan Cold Start<b> - Jika tidak ada riwayat tetapi ada genre favorit => Buat profil dari embedding teks genre favorit tersebut.<br>
            7. <b>Penanganan Cold Start<b> - Jika tidak ada riwayat dan tidak ada genre favorit → Rekomendasikan film terbaru berdasarkan <code>release_year</code>.<br>
            9. <b>Cosine Similarity</b> - hitung kemiripan profil user dengan semua film di katalog<br>
            10. <b>Filter & Ranking</b> - hilangkan film yang sudah ditonton, urutkan skor tertinggi<br>
            11. <b>Tampilkan Top-N</b> rekomendasi beserta skor kemiripan
        </div>
        """, unsafe_allow_html=True)