# version  noviembre 25, 2024
# update: 2024-07-28
# version final

import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.metric_cards import style_metric_cards
import numpy as np
import pandas as pd
from millify import millify
from funciones import menu_basic

st.set_page_config(
    page_title="Basic data",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ESTILOS
with open('style/style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(no_sidebar_style, unsafe_allow_html=True)
# --------------------------
# sidebar
# --------------------------
menu_basic()

# --------------------------
# main
# --------------------------

st.write("""
         # Análisis exploratorio de datos
         Se realizará un análisis exploratorio de los datos ingresados.
         """)
