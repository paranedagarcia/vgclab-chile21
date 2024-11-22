import pandas as pd
import numpy as np
import streamlit as st
from streamlit_extras.app_logo import add_logo


@st.cache_data
def load_data_csv(filename, chunks=False):
    # Load the data from the CSV file separated by comma or semicomma

    if filename is not None:
        try:
            data = pd.read_csv(filename, sep=",",
                               low_memory=False, index_col=0)
        except:
            data = pd.read_csv(filename, sep=";",
                               low_memory=False, index_col=0)
        return data


def logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(images/vgclab-negro.jpg);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            /*[data-testid="stSidebarNav"]::before {
                content: "My Company Name";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }*/
        </style>
        """,
        unsafe_allow_html=True,
    )


def menu_inicio():
    st.sidebar.page_link("Inicio.py", label=":house: Inicio")


@st.cache_data
def convert_df(df):
    csv = df.to_csv(sep=",", encoding='utf-8', index=False)
    return csv


@st.cache_data
def menu_pages():
    st.sidebar.image("images/vgclab-negro.jpg")
    st.sidebar.page_link("Inicio.py", label="Inicio")
    st.sidebar.page_link("pages/Conflictos_mapuches.py",
                         label="Violencia etnopolítica")
    st.sidebar.page_link("pages/Conflictos_sociales.py",
                         label="Conflictos sociales")
    st.sidebar.page_link("pages/Detenciones_por_delitos.py",
                         label="Detenciones por delitos")
    st.sidebar.page_link("pages/Salud_mental.py",
                         label="Salud mental (licencias)")
    # st.sidebar.page_link("pages/Salud_mental_full.py",
    #                     label=":bar_chart: Salud mental")
    st.sidebar.page_link("pages/Analisis.py",
                         label="Análisis comparativo")
    st.sidebar.page_link("pages/Analisis_par.py",
                         label="Análisis pareado")
    # st.sidebar.page_link("pages/ia.py",
    #                     label=":bar_chart: Análisis exploratorio con IA")
    # st.sidebar.page_link("pages/openai.py",
    #                      label=":bar_chart: IA con OpenAI")
    # st.sidebar.page_link("pages/langchain.py",
    #                      label=":bar_chart: IA con Langchain")
    # st.sidebar.page_link("pages/langchain_db.py",
    #                      label=":bar_chart: IA con Langchain DB")
    st.sidebar.page_link("pages/langchain_data.py",
                         label="Exploratorio con IA")