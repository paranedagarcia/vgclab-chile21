# version  julio 28, 2024
# update: 2024-07-28
# version final
import streamlit as st
import numpy as np
import pandas as pd
import base64

from funciones import menu_pages, logo, load_data_csv
# Page title
# configuration
st.set_page_config(
    page_title="VGCLAB",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ESTILOS
with open('style/style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# page_bg_img = '''
# <style>
# body {
# background-image: url("https://vgclabfiles.blob.core.windows.net/public/databox.jpg");
# background-size: cover;
# }
# </style>
# '''

# st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://vgclabfiles.blob.core.windows.net/public/databox.jpg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)


logo()

no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(no_sidebar_style, unsafe_allow_html=True)

# --------------------------
# sidebar
# --------------------------
# st.sidebar.image(image)

# pages
menu_pages()

st.sidebar.text("")
st.sidebar.write("[Fundación Chile 21](https://chile21.cl/)")
st.sidebar.divider()
st.sidebar.write("San Sebastián 2807, \n\rLas Condes, \nSantiago de Chile")


# st.header("Fundación Chile 21")
# st.subheader("Plataforma de datos")
# st.divider()


col1, col2 = st.columns([2, 2], gap="medium")

# with col1:
#     image = 'images/maceda.jpg'
#     st.image(image)
# with col2:
st.write("""<div class='content'><p>El Laboratorio sobre Violencias y Gestión de Conflictos VGC LAB es un espacio de reflexión, investigación colaborativa y generación de propuestas de intervención social para abordar la violencia y los conflictos en diferentes ámbitos, con el fin de promover una cultura de la paz y buena gestión de los conflictos.</p></div>""", unsafe_allow_html=True)
