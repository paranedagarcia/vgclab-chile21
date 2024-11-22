'''
Analisis entre datasets
Fecha: 2024-06-06
Autor: Patricio Araneda
'''

# librerias
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import os

import pandas as pd
import numpy as np
import time
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from funciones import menu_pages
import duckdb
# import statsmodels.api as sm
import seaborn as sns
# import ace_tools as tools

# configuration
st.set_page_config(
    page_title="Análisis",
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


menu_pages()

# --------------------------
# MAIN
# --------------------------

st.subheader("Análisis pareado de datasets")


dataframes = ['Maceda', 'Conflictos', 'Detenciones', 'Salud']

datasets = {'Seleccione': None,
            'Maceda': 'maceda',
            'Conflictos sociales': 'conflictos',
            'Detenciones': 'detenciones',
            'Licencias médicas': 'licencias'}

tipoanalisis = {'Seleccione análisis': None,
                'Comuna': 'comuna',
                'Provincia': 'provincia',
                'Región': 'region',
                'Fecha': 'fecha'}

# MAIN


@st.cache_data
def corr_data(ds1, ds2, tipo):
    conn = duckdb.connect('vglab.db')
    df1 = conn.execute(f"SELECT * FROM {ds1}").fetchdf()
    df2 = conn.execute(f"SELECT * FROM {ds2}").fetchdf()
    conn.close()
    df1.columns = df1.columns.str.lower()
    df2.columns = df2.columns.str.lower()
    # limpieza para licencias
    if ds1 == 'licencias':
        df1['fecha'] = pd.to_datetime(df1['emision'])
    if ds2 == 'licencias':
        df2['fecha'] = pd.to_datetime(df2['emision'])

    # st.write(df1.head(), df2.head())
    if tipo == 'fecha':
        if ds1 != 'detenciones':
            dc1 = df1.groupby(df1['fecha'].dt.to_period('M')
                              ).size().reset_index(name=f'{ds1}')
        else:
            dc1 = df1.groupby(df1['fecha'].dt.to_period('M')
                              )['casos'].sum().reset_index(name=f'{ds1}')
        if ds2 != 'detenciones':
            dc2 = df2.groupby(df2['fecha'].dt.to_period('M')
                              ).size().reset_index(name=f'{ds2}')
        else:
            dc2 = df2.groupby(df2['fecha'].dt.to_period('M')
                              )['casos'].sum().reset_index(name=f'{ds2}')
    else:
        tipo = f'codigo{tipo}'
        if ds1 != 'detenciones':
            dc1 = df1.groupby(f'{tipo}').size().reset_index(name=f'{ds1}')
        else:
            dc1 = df1.groupby(f'{tipo}')[
                'casos'].sum().reset_index(name=f'{ds1}')
        if ds2 != 'detenciones':
            dc2 = df2.groupby(f'{tipo}').size().reset_index(name=f'{ds2}')
        else:
            dc2 = df2.groupby(f'{tipo}')[
                'casos'].sum().reset_index(name=f'{ds2}')

    combined = pd.merge(dc1, dc2, on=[
        f'{tipo}'], how='inner')
    # c_corr = combined[dc1].corr(combined[dc2])
    # st.write(df1.head(), df2.head(), combined.head())

    def get_key(val):
        key_ds1 = [key for key, value in datasets.items() if value == val][0]
        if key_ds1:
            return key_ds1
        else:
            return None
    # st.write(get_key(ds1), get_key(ds2))

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=f'{ds1}', y=f'{ds2}',
                    data=combined)

    sns.regplot(x=f'{ds1}', y=f'{ds2}',
                data=combined, scatter=False, color='red')

    plt.title(f'{get_key(ds1)} y {get_key(ds2)} ', fontsize=12)
    plt.xlabel(f'{ds1}', fontsize=10)
    plt.ylabel(f'{ds2}', fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    st.pyplot(plt)

    # return combined


st.write("Comparación de datasets, elija dos datasets y el tipo de análisis. Licencias Médicas no tiene datos de locación, por lo que no se puede comparar comuna, provincia o región.")

col1, col2 = st.columns([1, 3], gap="small")
with col1:
    with st.form(key="my_form"):

        ds1_options = list(datasets.keys())
        dataset1 = st.selectbox(
            'Seleccione dataset 1', options=ds1_options, index=1)

        ds2_options = list(datasets.keys())
        dataset2 = st.selectbox(
            'Seleccione dataset 2', options=ds1_options, index=2)

        tipo = st.selectbox('Seleccione tipo de análisis',
                            options=tipoanalisis, index=1)

        submitted = st.form_submit_button("Comparar datasets")

with col2:
    if submitted:
        ds1 = datasets[dataset1]
        ds2 = datasets[dataset2]
        tipo = tipoanalisis[tipo]

        # st.write(ds1, ds2, tipo)

        corr_data(ds1, ds2, tipo)
