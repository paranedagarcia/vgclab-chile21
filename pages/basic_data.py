# version  noviembre 25, 2024
# update: 2024-07-28
# version final

import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.metric_cards import style_metric_cards
import numpy as np
import pandas as pd
from millify import millify
from funciones import menu_basic, menu_pages, logo
from pygwalker.api.streamlit import StreamlitRenderer
import plotly.express as px

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
tabPanel, tabTable, tabEda, tabCorr, tableau, tabGeo = st.tabs(
    ["Data", "Tabla", "EDA", "Correlaciones", "Visualización", "Geodata"])

with tabPanel:  # data

    st.write("""
            ## Basic data
            Ingrese su archivo de datos en formato CSV y seleccione los análisis que desee aplicar. Se realizará un análisis exploratorio de los datos.
            """)

    mifile = st.file_uploader("Subir archivo CSV", type=["csv"])

    if mifile is not None:
        df = pd.read_csv(mifile)

        # Identificar columnas categóricas y numéricas
        categorical_columns = df.select_dtypes(
            include=['object']).columns.tolist()
        numerical_columns = df.select_dtypes(
            include=['number']).columns.tolist()

        # Mostrar valores en tarjetas
        st.write("#### Resumen de datos")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total de registros",
                      value=millify(len(df), precision=2))
        with col2:
            st.metric(label="Columnas categóricas",
                      value=len(categorical_columns))
        with col3:
            st.metric(label="Columnas numéricas", value=len(numerical_columns))
        with col4:
            st.metric(label="Valores nulos", value=millify(
                df.isnull().sum().sum(), precision=2))

        style_metric_cards()

        st.write("#### Primeros 10 registros del archivo")
        st.write(df.head(10))

        col1, col2 = st.columns(2)
        with col1:
            categorical = st.multiselect(
                "Columnas categóricas", options=categorical_columns, placeholder="Columnas categóricas", max_selections=10)
        with col2:
            numerical = st.multiselect(
                "Columnas numéricas", options=numerical_columns, placeholder="Columnas numéricas", max_selections=10)

        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Estadísticas descriptivas")
            st.write(df.describe())

        with col2:
            st.write("#### Resumen de datos")
            st.write(df.info())

        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Valores nulos")
            st.write(df.isnull().sum())

        with col2:
            st.write("#### Valores duplicados")
            st.write(df.duplicated().sum())

# --------------------------
with tabTable:  # tabla de datos
    if mifile is not None:
        st.dataframe(df, height=600)

with tabEda:  # eda
    st.write("""
            ## Análisis exploratorio de datos""")
    if mifile is not None:
        st.write("Seleccione las columnas que desea explorar.")

        # --------------------------
        # Columnas a eliminar por cantidad de nulos
        # --------------------------
        with st.form(key="nulos", border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                nulos = st.number_input("Límite de nulos", value=1, min_value=1,
                                        max_value=500000, placeholder="Cantidad de nulos")
            with col2:
                if nulos is not 0:
                    columnas_nulos = [
                        col for col in df.columns if df[col].isnull().sum() > nulos]
                    columnas_nulos.sort()

                col_elim = st.multiselect(
                    "Columnas a eliminar", options=columnas_nulos, placeholder="Columnas a eliminar")

            with col3:
                anulados = st.form_submit_button("Eliminar columnas")

            if anulados:
                df = df.drop(columns=col_elim)

        # Calcular la cantidad de valores nulos por columna
        null_counts = df.isnull().sum()
        # Crear el gráfico de barras
        fig = px.bar(null_counts,
                     title="Cantidad de valores nulos por columna",
                     color_discrete_sequence=px.colors.qualitative.Prism,
                     labels={
                         'index': 'Columna', 'value': 'Valores nulos'})
        fig.update_layout(showlegend=False, height=600,
                          title={
                              'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig, use_container_width=True)

        st.write(df.head(10))

    else:
        st.write("Suba un archivo CSV para continuar.")

with tabCorr:  # corr
    st.write("""
            ## Correlaciones básicas
            Explore las posibles correlaciones entre sus datos.
            """)
with tableau:  # tableau
    st.write("""
            ## Visualización de datos
            """)
    if mifile is not None:
        pyg_app = StreamlitRenderer(df, spec_io_mode="rw")
        pyg_app.explorer()

with tabGeo:  # geo
    st.write("""
            ## Geolocalización
            Si sus datos contiene información geográfica. Ya sea tenga información de latitud y longitud o de nombres de comunas, podrá visualizarla en un mapa interactivo.
            """)
    if mifile is not None:
        pyg_app = StreamlitRenderer(df, spec_io_mode="rw")
        pyg_app.explorer()
