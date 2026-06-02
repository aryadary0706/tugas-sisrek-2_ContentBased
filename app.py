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
# ─────────────────────────────────────────────
#  SYNTHETIC USER DATA (12 Variasi Kasus Ekstrem)
# ─────────────────────────────────────────────
USERS = {
    # 1. Genre Terdaftar, History Lumayan, Semua Judul Ada di Dataset (Normal - High Score)
    "U001": {
        "name": "Arya Kusuma",
        "age": 24,
        "preferred_genres": ["Thrillers", "Sci-Fi & Fantasy"],
        "watch_history": [
            {"title": "Bird Box"},
            {"title": "The Platform"},
            {"title": "Black Mirror: Bandersnatch"},
            {"title": "Inception"},
            {"title": "Annihilation"},
        ]
    },
    # 2. Genre Terdaftar, History Sedikit, Semua Judul Ada di Dataset
    "U002": {
        "name": "Siti Rahayu",
        "age": 31,
        "preferred_genres": ["Romantic Movies", "Comedies"],
        "watch_history": [
            {"title": "To All the Boys I've Loved Before"},
            {"title": "Always Be My Maybe"},
        ]
    },
    # 3. Genre Terdaftar, Sebagian History Ada di Dataset (Campuran Film Indo Lama & Baru)
    "U003": {
        "name": "Budi Santoso",
        "age": 28,
        "preferred_genres": ["Dramas", "International Movies"],
        "watch_history": [
            {"title": "Gie"},                       # Ada di Dataset Netflix (Lama)
            {"title": "Srimulat: Hil yang Mustahal"},# TIdak Ada di Dataset Netflix (Baru)
            {"title": "Ali & Ratu Ratu Queens"},     # Ada di Dataset Netflix
            {"title": "Agak Laen"},                 # Tidak Ada di Dataset Netflix (Baru)
        ]
    },
    # 4. Genre Terdaftar, TAPI Seluruh History TIDAK Ada di Dataset (Film Indo Bioskop Terbaru)
    # Efek: Cold start terpicu (Fallback ke Preferred Genres), kemiripan genre mungkin masih tinggi, tapi hit rate bisa jatuh jika rekomendasinya meleset dari history aktor/sutradara.
    "U004": {
        "name": "Dewi Lestari",
        "age": 27,
        "preferred_genres": ["Horror Movies", "Thrillers"],
        "watch_history": [
            {"title": "Siksa Kubur"},               # Tidak Ada (Film 2024)
            {"title": "KKN di Desa Penari"},        # Tidak Ada
            {"title": "Pengabdi Setan 2: Communion"},# Tidak Ada
        ]
    },
    # 5. Genre Sebagian Ada (Campuran Normal & Absurd), History Watch Banyak (Ada di Dataset)
    "U005": {
        "name": "Reza Firmansyah",
        "age": 19,
        "preferred_genres": ["Action & Adventure", "Otomotif Karburator Honda"], # Absurd campuran
        "watch_history": [
            {"title": "John Wick"},
            {"title": "Extraction"},
            {"title": "The Old Guard"},
            {"title": "6 Underground"},
        ]
    },
    # 6. Genre Sebagian Ada (Campuran), History Watch Sedikit (Ada di Dataset)
    "U006": {
        "name": "Nia Permata",
        "age": 35,
        "preferred_genres": ["Documentaries", "Kriptografi Quantum Cyber"], # Absurd campuran
        "watch_history": [
            {"title": "Our Planet"},
            {"title": "Making a Murderer"},
        ]
    },
    # 7. Genre Sebagian Ada, History Watch Banyak TAPI TIDAK ADA di Dataset
    "U007": {
        "name": "Farhan Nugroho",
        "age": 29,
        "preferred_genres": ["Comedies", "Budidaya Ikan Lele Kolam Terpal"], 
        "watch_history": [
            {"title": "Ancika: Dia yang Bersamaku 1995"}, # Tidak Ada
            {"title": "Petualangan Sherina 2"},          # Tidak Ada
            {"title": "Pasutri Gaje"},                   # Tidak Ada
            {"title": "Kaka Boss"},                      # Tidak Ada
        ]
    },
    # 8. Genre Sebagian Ada, History Watch Sedikit TAPI TIDAK ADA di Dataset
    "U008": {
        "name": "Maya Indah",
        "age": 22,
        "preferred_genres": ["Children & Family Movies", "Resep Seblak Ceker Pedas"], 
        "watch_history": [
            {"title": "Badarawuhi di Desa Penari"}, # Tidak Ada
            {"title": "Vina: Sebelum 7 Hari"},       # Tidak Ada
        ]
    },
    # 9. TARGET METRIK RENDAH: Genre Semuanya TIdak Ada (Sangat Absurd), History Watch Banyak & Terdaftar
    # Efek: Mengacaukan profil agregasi rata-rata vektor jika model dipaksa membaca keyword absurd.
    "U009": {
        "name": "Hendra Wijaya",
        "age": 45,
        "preferred_genres": ["Mesin Jahit Konveksi", "Suku Cadang Mesin Diesel Diesel"],
        "watch_history": [
            {"title": "Inception"},
            {"title": "Interstellar"},
            {"title": "The Matrix"},
            {"title": "Blade Runner 2049"},
        ]
    },
    # 10. TARGET METRIK RENDAH: Genre Semuanya Tidak Ada (Absurd), History Watch Sedikit & Terdaftar
    "U010": {
        "name": "Ratna Sari",
        "age": 33,
        "preferred_genres": ["Pertanian Organik Hidroponik", "Tekstil Industri Kain Katun"],
        "watch_history": [
            {"title": "The Notebook"},
            {"title": "Pride & Prejudice"},
        ]
    },
    # 11. PENGHANCUR METRIK (TARGET HIT RATE = 0 & SIMILARITY < 0.4): 
    # Genre Semuanya Tidak Ada (Absurd), History Banyak TAPI SEMUANYA TIDAK ADA di Dataset.
    # Efek: Sistem terpaksa masuk ke mode Cold-Start murni memakai teks absurd. SBERT akan menghasilkan similarity score yang sangat rendah (< 0.4) dengan katalog film bioskop, dan evaluasi Overlap pastinya bernilai 0 (Hit Rate = 0).
    "U011": {
        "name": "Dimas Prasetyo",
        "age": 25,
        "preferred_genres": ["Alat Pertukangan Semen Semprot", "Sistem Pipa Pembuangan Lumpur Sidoarjo"],
        "watch_history": [
            {"title": "Sekawan Limo"},            # Tidak Ada
            {"title": "Ipar adalah Maut"},         # Tidak Ada
            {"title": "Jurnal Risa by Risa Saraswati"}, # Tidak Ada
            {"title": "Do You See What I See"},    # Tidak Ada
        ]
    },
    # 12. PENGHANCUR METRIK: Genre Semuanya Tidak Ada (Absurd), History Sedikit & TIDAK ADA di Dataset
    "U012": {
        "name": "Laila Azzahra",
        "age": 30,
        "preferred_genres": ["Manajemen Akuntansi Neraca Saldo", "Kalkulus Integral Turunan Parsial"],
        "watch_history": [
            {"title": "Kang Mak from Pee Mak"}, # Tidak Ada
            {"title": "Bolehkah Sekali Ini Saja Menangis"}, # Tidak Ada
        ]
    }
}

# ─────────────────────────────────────────────
#  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

def clean_names(text):
    if isinstance(text, str):
        text = text.lower()
        names = text.split(',')
        return " ".join([n.replace(" ", "") for n in names])
    return ""
    
def clean_genres(genre_string):
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
    
    # Gabungkan semua fitur menjadi satu string untuk setiap film
    df['cast_cleaned'] = df['cast'].apply(clean_names)
    df['director_cleaned'] = df['director'].apply(clean_names)
    df['listed_in_cleaned'] = df['listed_in'].apply(clean_genres)
    def combine_features(row):
        return " ".join([str(row['title']), str(row['rating']), " ".join(row['listed_in_cleaned']), str(row['description']), str(row['director_cleaned']), str(row['cast_cleaned'])])

    df['combined_features'] = df.apply(combine_features, axis=1).str.lower()
    return df

@st.cache_resource(show_spinner="🤖 Membuat SBERT embeddings (tunggu sebentar)...")
def build_embeddings(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
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
def get_recommendations(user_data, df_catalog, embeddings_matrix, top_n=5):
    age = user_data.get('age', 18)
    watch_history = user_data.get('watch_history', [])
    preferred_genres = user_data.get('preferred_genres', [])
    
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
            pref_text = " ".join(clean_user_pref)
            user_profile_vector = model.encode([pref_text]).reshape(1, -1)
            scores = cosine_similarity(user_profile_vector, filtered_embeddings)[0]
        else:
            scores = np.full(len(df_scores), 0.1000)
    else:
        user_vecs = filtered_embeddings[history_indices_in_filtered]
        user_profile_vector = np.mean(user_vecs, axis=0).reshape(1, -1)
        scores = cosine_similarity(user_profile_vector, filtered_embeddings)[0]

    df_scores['similarity_score'] = scores
    
    df_final = df_scores[~df_scores['title'].str.lower().isin(history_titles)]
    return df_final.sort_values('similarity_score', ascending=False).head(top_n)

# 5. Fungsi Evaluasi Metrik
def evaluate_recommendations(recommendations, user_data, df_catalog):
    if recommendations.empty:
        return {"avg_similarity": 0, "f1_score": 0, "hit_rate": 0}
    
    avg_sim = recommendations['similarity_score'].mean()
    
    # Masukkan target kata dari preferred_genres
    user_target_words = set()
    for g in user_data.get('preferred_genres', []):
        user_target_words.update(clean_genres(g))
    
    # TAMBAHAN: Ekstrak juga director, cast, dan genre dari watch history user
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
    
    for _, row in recommendations.iterrows():
        # Gabungkan kata dari genre, director, dan cast pada film rekomendasi untuk dicek kecocokannya
        rec_words = set(clean_genres(row['listed_in']))
        if row['director']:
            rec_words.update(row['director'].lower().replace(',', ' ').split())
        if row['cast']:
            rec_words.update(row['cast'].lower().replace(',', ' ').split())
        
        intersection = user_target_words.intersection(rec_words)
        
        if len(intersection) > 0:
            hits += 1
            
        if len(rec_words) == 0 or len(user_target_words) == 0:
            f1 = 0
        else:
            precision = len(intersection) / len(rec_words)
            recall = len(intersection) / len(user_target_words)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        
    hit_rate = 1 if hits > 0 else 0
    avg_f1 = np.mean(f1_scores)
    
    return {
        "avg_similarity": round(float(avg_sim), 4),
        "f1_score": round(float(avg_f1), 4),
        "hit_rate": hit_rate
    }


# ─────────────────────────────────────────────
#  SIDEBAR – User Selection & Info
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Pilih Active User")

    user_labels = {uid: f"{uid} - {info['name']}" for uid, info in USERS.items()}
    selected_uid = st.selectbox(
        "User ID",
        options=list(USERS.keys()),
        format_func=lambda uid: user_labels[uid]
    )

    top_n = st.slider("Jumlah rekomendasi", min_value=3, max_value=8, value=5)

    st.markdown("---")
    st.markdown("**Daftar semua User ID:**")
    for uid, info in USERS.items():
        marker = "🟥" if uid == selected_uid else "⚪"
        st.markdown(f"{marker} `{uid}` {info['name']}")


# ─────────────────────────────────────────────
#  LOAD DATA  &  EMBEDDINGS
# ─────────────────────────────────────────────

# Cache data dan embeddings agar tidak perlu dihitung ulang setiap interaksi user
df = load_data()

# Ambil data user yang dipilih
user = USERS[selected_uid]

# Bangun matriks embedding untuk seluruh katalog (sekali saja, cached)
embeddings = build_embeddings(df['combined_features'].tolist())

# ─────────────────────────────────────────────
#  FUNGSI REKOMENDASI & EVALUASI
# ─────────────────────────────────────────────
reccommendations_df = get_recommendations(user, df, embeddings, top_n=top_n)
metrics = evaluate_recommendations(reccommendations_df, user, df)

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
            
            st.markdown(f"""
            <div class="user-card" style="padding:10px 16px;">
              🎥 <b>{movie['title']}</b><br>
              <span style="color:#888; font-size:.78rem;"> Rating Konten: {rating_content}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-title">🔎 Rekomendasi Konten</div>', unsafe_allow_html=True)

        # Cek apakah hasil rekomendasi kosong
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


# ────────── TAB 2 : EVALUASI ──────────
with tab2:
    st.markdown('<div class="section-title">📐 Metrik Evaluasi untuk User Aktif</div>', unsafe_allow_html=True)
    st.info("Evaluasi menggunakan **genre, director, dan cast overlap** sebagai *ground truth* untuk mengukur relevansi konten yang direkomendasikan.")

    # Gunakan variabel 'metrics' dan 'recommendations_df' yang sudah dihitung di luar tab
    if metrics and not reccommendations_df.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-box">
              <div class="val">{metrics['hit_rate']}</div>
              <div class="lbl">Hit Rate (Akurasi Rekomendasi)</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-box">
              <div class="val">{metrics['avg_similarity']:.4f}</div>
              <div class="lbl">Avg Cosine Similarity</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-box">
              <div class="val">{metrics['f1_score']:.4f}</div>
              <div class="lbl">Mean F1-Score (Overlap)</div></div>""", unsafe_allow_html=True)
    else:
        st.warning("Tidak dapat menghitung metrik - Data rekomendasi kosong atau riwayat user tidak ditemukan di katalog.")

    st.markdown("---")
    st.markdown('<div class="section-title">📖 Penjelasan Metrik</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="info-box">
          <h4>🎯 Hit Rate</h4>
          Apakah sistem berhasil memberikan **minimal 1 item** yang relevan dari Top-N rekomendasi?<br><br>
          <code>1 = Sukses (Ada Overlap)</code><br>
          <code>0 = Gagal (Tidak ada Overlap sama sekali)</code>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="info-box">
          <h4>📡 Avg Cosine Similarity</h4>
          Mengukur seberapa dekat secara semantik (*SBERT embedding*) item yang direkomendasikan dengan preferensi gabungan user.<br><br>
          Semakin mendekati 1.0, berarti kualitas kemiripan teks/sinopsis film semakin tinggi.
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="info-box">
          <h4>⚖️ F1-Score (Feature Overlap)</h4>
          Rata-rata harmonik antara presisi dan recall dari kemunculan kata kunci target (genre, sutradara, aktor) di film rekomendasi dibandingkan dengan riwayat tontonan user.
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