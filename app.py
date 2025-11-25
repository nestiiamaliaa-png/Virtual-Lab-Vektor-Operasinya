import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ============================
#   HEADER APLIKASI
# ============================

st.set_page_config(page_title="Virtual Lab: Vektor & Operasinya", layout="wide")

st.title("🧮 Virtual Lab: Vektor & Operasinya")
st.markdown("""
Lab interaktif untuk mempelajari operasi vektor:
- ➕ Penjumlahan vektor  
- ✖️ Perkalian skalar  
- 🔵 Dot Product  
- 🔁 Cross Product  
- 📊 Visualisasi 2D/3D  
""")

st.divider()

# ============================
#   INPUT VEKTOR
# ============================

st.sidebar.header("Input Vektor")

v1 = st.sidebar.text_input("Masukkan Vektor A (pisahkan dengan koma)", "2, 1, 0")
v2 = st.sidebar.text_input("Masukkan Vektor B (pisahkan dengan koma)", "1, -1, 3")

try:
    A = np.array([float(x) for x in v1.split(",")])
    B = np.array([float(x) for x in v2.split(",")])
except:
    st.error("Format input salah! Contoh input: 1, 2, 3")
    st.stop()

dim = len(A)

# ============================
#   PILIH OPERASI
# ============================

operasi = st.selectbox(
    "Pilih Operasi Vektor",
    [
        "Penjumlahan Vektor (A + B)",
        "Perkalian Skalar (k × A)",
        "Dot Product (A · B)",
        "Cross Product (A × B)",
        "Visualisasi 2D",
        "Visualisasi 3D"
    ]
)

# ============================
#   HASIL OPERASI
# ============================

st.subheader("📌 Hasil Perhitungan")

if operasi == "Penjumlahan Vektor (A + B)":
    if len(A) != len(B):
        st.error("Dimensi vektor harus sama!")
    else:
        result = A + B
        st.markdown(f"**A + B = {result}**")


elif operasi == "Perkalian Skalar (k × A)":
    k = st.number_input("Masukkan skalar k:", value=2.0)
    result = k * A
    st.markdown(f"**{k} × A = {result}**")


elif operasi == "Dot Product (A · B)":
    if len(A) != len(B):
        st.error("Dimensi vektor harus sama!")
    else:
        result = np.dot(A, B)
        st.markdown(f"**A · B = {result}**")


elif operasi == "Cross Product (A × B)":
    if len(A) != 3 or len(B) != 3:
        st.error("Cross product hanya berlaku untuk vektor 3D!")
    else:
        result = np.cross(A, B)
        st.markdown(f"**A × B = {result}**")


# ============================
#   VISUALISASI 2D
# ============================

elif operasi == "Visualisasi 2D":
    if dim != 2:
        st.error("Visualisasi 2D hanya untuk vektor dimensi 2!")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, A[0]], y=[0, A[1]], mode="lines+markers", name="A"))
        fig.add_trace(go.Scatter(x=[0, B[0]], y=[0, B[1]], mode="lines+markers", name="B"))

        fig.update_layout(
            width=700, height=500,
            title="Visualisasi Vektor 2D",
            xaxis_title="X",
            yaxis_title="Y",
            showlegend=True
        )
        st.plotly_chart(fig)


# ============================
#   VISUALISASI 3D
# ============================

elif operasi == "Visualisasi 3D":
    if dim != 3:
        st.error("Visualisasi 3D hanya untuk vektor dimensi 3!")
    else:
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=[0, A[0]],
            y=[0, A[1]],
            z=[0, A[2]],
            mode="lines+markers",
            name="A"
        ))

        fig.add_trace(go.Scatter3d(
            x=[0, B[0]],
            y=[0, B[1]],
            z=[0, B[2]],
            mode="lines+markers",
            name="B"
        ))

        fig.update_layout(
            width=800, height=600,
            title="Visualisasi Vektor 3D",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            )
        )

        st.plotly_chart(fig)

# ============================
#   TAMPILKAN VEKTOR AWAL
# ============================

st.divider()
st.subheader("📌 Vektor yang Digunakan")

st.markdown(f"**A = {A}**")
st.markdown(f"**B = {B}**")
