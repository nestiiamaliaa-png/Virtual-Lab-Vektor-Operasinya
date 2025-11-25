# Virtual Lab: Vektor & Operasinya
# File: virtual_lab_vektor_streamlit.py
# Deskripsi: Aplikasi Streamlit interaktif untuk memvisualisasikan dan
# menjelaskan operasi vektor: penjumlahan, perkalian skalar, dot product,
# cross product (untuk 3D), dan visualisasi 2D/3D. Siswa dapat mengubah
# vektor lewat input numerik/sliders, melihat langkah komputasi, matriks,
# sudut antar vektor, proyeksi, serta mengunduh data koordinat.

"""
Instruksi singkat:
- Jalankan lokal: python -m pip install -r requirements.txt
  lalu: streamlit run virtual_lab_vektor_streamlit.py
- Untuk deploy lewat GitHub: buat repo baru, push file ini + requirements.txt,
  lalu gunakan Streamlit Cloud (https://share.streamlit.io) untuk menghubungkan repo.

requirements.txt (tambahkan file ini di repo):
streamlit
numpy
pandas
matplotlib
plotly
scipy

Catatan: file ini adalah aplikasi tunggal - ubah sesuai preferensi.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import plotly.graph_objects as go
from io import StringIO
from math import acos, degrees
from scipy.spatial.transform import Rotation as R

st.set_page_config(page_title="Virtual Lab: Vektor & Operasinya", layout="wide")

# --- Helper functions ---

def parse_vector(text, dim=3):
    try:
        parts = [float(x) for x in text.replace(',', ' ').split()]
        if len(parts) == dim:
            return np.array(parts)
        elif len(parts) < dim:
            parts += [0.0] * (dim - len(parts))
            return np.array(parts)
        else:
            return np.array(parts[:dim])
    except Exception:
        return np.zeros(dim)


def angle_between(u, v):
    uu = np.linalg.norm(u)
    vv = np.linalg.norm(v)
    if uu == 0 or vv == 0:
        return None
    cosang = np.dot(u, v) / (uu * vv)
    cosang = max(min(cosang, 1.0), -1.0)
    return degrees(acos(cosang))


def projection(u, v):
    # projection of u onto v
    vv = np.dot(v, v)
    if vv == 0:
        return np.zeros_like(u)
    return (np.dot(u, v) / vv) * v


def to_dataframe(vectors, names):
    df = pd.DataFrame(vectors, columns=[f"x{i+1}" for i in range(vectors.shape[1])])
    df.insert(0, 'name', names)
    return df

# --- Sidebar: pilihan umum ---
st.sidebar.title("Pengaturan Virtual Lab")
dim_choice = st.sidebar.radio("Dimensi visualisasi", ('2D', '3D'))
show_steps = st.sidebar.checkbox("Tampilkan langkah perhitungan", value=True)
show_matrix = st.sidebar.checkbox("Tampilkan matriks transformasi (jika ada)", value=False)

st.title("🔷 Virtual Lab: Vektor & Operasinya")
st.markdown("Pelajari penjumlahan vektor, perkalian skalar, dot & cross product, dan visualisasi 2D/3D secara interaktif.")

# --- Input vektor ---
st.header("Input Vektor")
col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Masukkan dua vektor")
    dim = 2 if dim_choice == '2D' else 3
    v1_text = st.text_input('Vektor A (pisahkan dengan spasi, contoh: 2 3 1)', '2 1 0' if dim==3 else '2 1')
    v2_text = st.text_input('Vektor B (pisahkan dengan spasi)', ' -1 2 1' if dim==3 else '-1 2')
    v1 = parse_vector(v1_text, dim)
    v2 = parse_vector(v2_text, dim)

    st.markdown("**Opsi lain:**")
    with st.expander("Buat vektor acak / reset"):
        if st.button('Acak vektor (kecil)'):
            v1 = (np.random.randint(-5,6,size=dim)).astype(float)
            v2 = (np.random.randint(-5,6,size=dim)).astype(float)
            st.experimental_rerun()

with col2:
    st.subheader("Kontrol interaktif")
    factor = st.slider('Skalar untuk perkalian (alpha)', -5.0, 5.0, 2.0, 0.1)
    show_grid = st.checkbox('Tampilkan grid pada grafik', value=True)

# --- Operasi ---
st.header("Operasi Vektor")
colA, colB, colC = st.columns(3)
with colA:
    st.subheader('Penjumlahan')
    sum_v = v1 + v2
    st.write('A + B =', sum_v)
    if show_steps:
        st.markdown(f"A = {v1}  
B = {v2}  \nA + B = {v1} + {v2} = {sum_v}")

with colB:
    st.subheader('Perkalian Skalar')
    scalar_v = factor * v1
    st.write(f'{factor} * A =', scalar_v)
    if show_steps:
        st.markdown(f"{factor} * A = {factor} * {v1} = {scalar_v}")

with colC:
    st.subheader('Dot Product')
    dot = float(np.dot(v1, v2))
    st.write('A · B =', dot)
    ang = angle_between(v1, v2)
    if ang is not None:
        st.write('Sudut antara A dan B ≈', f"{ang:.2f}°")
    if show_steps:
        st.markdown(f"A · B = sum(A_i * B_i) = {dot}")

if dim == 3:
    st.subheader('Cross Product (hanya 3D)')
    cross = np.cross(v1, v2)
    st.write('A × B =', cross)
    if show_steps:
        st.markdown(f"A × B = {cross} (vektor yang tegak lurus pada A dan B)")

# --- Visualisasi ---
st.header('Visualisasi')
if dim_choice == '2D':
    st.subheader('Plot 2D (matplotlib)')
    fig, ax = plt.subplots(figsize=(6,6))
    max_range = max(np.max(np.abs(np.vstack([v1, v2, sum_v, scalar_v]))), 1) + 1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    if show_grid:
        ax.grid(True, which='both')

    def draw_vec(ax, vec, color='k', label=None, lw=2, alpha=1.0):
        ax.arrow(0,0,vec[0],vec[1],head_width=0.25, head_length=0.3, length_includes_head=True, linewidth=lw, alpha=alpha)
        if label:
            ax.text(vec[0]*1.05, vec[1]*1.05, label)

    draw_vec(ax, v1[:2], label='A')
    draw_vec(ax, v2[:2], label='B')
    draw_vec(ax, sum_v[:2], label='A+B', color='g', lw=2)
    draw_vec(ax, scalar_v[:2], label=f'{factor}A', color='m', lw=2, alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    st.pyplot(fig)

else:
    st.subheader('Plot 3D (plotly)')
    fig3 = go.Figure()
    origin = np.zeros(3)

    def add_vector_plot(fig, vec, name, color):
        fig.add_trace(go.Scatter3d(x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]], mode='lines+markers', marker=dict(size=3), name=name))

    add_vector_plot(fig3, v1, 'A', 'blue')
    add_vector_plot(fig3, v2, 'B', 'red')
    add_vector_plot(fig3, sum_v, 'A+B', 'green')
    add_vector_plot(fig3, scalar_v, f'{factor}A', 'purple')

    all_pts = np.vstack([v1, v2, sum_v, scalar_v])
    max_range = max(np.max(np.abs(all_pts)), 1) + 1
    fig3.update_layout(scene=dict(xaxis=dict(range=[-max_range, max_range]),
                                  yaxis=dict(range=[-max_range, max_range]),
                                  zaxis=dict(range=[-max_range, max_range])), width=700, height=700)
    st.plotly_chart(fig3, use_container_width=True)

# --- Proyeksi dan komponen ---
st.header('Proyeksi & Komponen')
proj = projection(v1, v2)
st.write('Proyeksi A pada B =', proj)
if show_steps:
    st.markdown(f"Proyeksi(A pada B) = (A·B / B·B) B = {proj}")

# --- Matriks transformasi contoh ---
if show_matrix:
    st.header('Contoh Matriks Transformasi pada Vektor 2D')
    st.markdown('Anda dapat menerapkan matriks transformasi sederhana ke vektor (rotasi, skala, refleksi).')
    theta = st.slider('Sudut rotasi (derajat)', -180, 180, 30)
    sx = st.number_input('Skala s_x', value=1.0, format='%f')
    sy = st.number_input('Skala s_y', value=1.0, format='%f')
    th = np.deg2rad(theta)
    rot = np.array([[np.cos(th), -np.sin(th), 0],[np.sin(th), np.cos(th), 0],[0,0,1]])
    scale = np.array([[sx,0,0],[0,sy,0],[0,0,1]])
    mat = scale @ rot
    st.write('Matriks 3x3 (homogen) =')
    st.write(mat)
    if dim == 2:
        v1h = np.append(v1[:2], 1)
        v2h = np.append(v2[:2], 1)
        tv1 = mat @ v1h
        tv2 = mat @ v2h
        st.write('Transformasi A =>', tv1[:2])
        st.write('Transformasi B =>', tv2[:2])

# --- Tabel & Download ---
st.header('Data & Unduh')
df = to_dataframe(np.vstack([v1, v2, sum_v, scalar_v, proj]) if dim==3 else np.vstack([v1[:2], v2[:2], sum_v[:2], scalar_v[:2], proj[:2]]),
                  ['A','B','A+B', f'{factor}A', 'proj(A on B)'])

st.dataframe(df)

csv = df.to_csv(index=False)
st.download_button('Unduh data sebagai CSV', csv, file_name='vektor_data.csv', mime='text/csv')

# --- Penjelasan singkat (didaktik) ---
st.header('Penjelasan Singkat')
st.markdown('''
- **Penjumlahan Vektor**: jumlah komponen sesuai posisi. Visualisasi memudahkan paham cara "tip-to-tail".
- **Perkalian Skalar**: mengubah panjang (dan arah jika negatif) vektor.
- **Dot Product**: ukuran proyeksi dan menentukan sudut antar vektor (skalar).
- **Cross Product** (3D): menghasilkan vektor yang tegak lurus pada kedua vektor (arah ditentukan aturan tangan kanan).

Gunakan slider, masukkan nilai, dan lihat bagaimana grafik serta angka berubah. Ajak siswa berhipotesis (mis. jika dot=0 apa yang terjadi?) lalu verifikasi dengan visual.
''')

st.info('Selesai — file aplikasi ada di repository Anda. Ingin saya tambahkan fitur latihan otomatis (quiz) atau mode langkah demi langkah penuh?')
