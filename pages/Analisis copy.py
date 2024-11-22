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

st.subheader("Análisis entre datasets")


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


progress_text = "Procesando datos..."
pro_bar = st.progress(0.20, text=progress_text)
# dataframes from duckdb
conn = duckdb.connect('vglab.db')
df1 = conn.execute("SELECT * FROM maceda").fetchdf()
pro_bar.progress(0.40, text=progress_text)
df2 = conn.execute("SELECT * FROM conflictos").fetchdf()
pro_bar.progress(0.60, text=progress_text)
df3 = conn.execute("SELECT * FROM detenciones").fetchdf()
pro_bar.progress(0.80, text=progress_text)
df4 = conn.execute("SELECT * FROM licencias").fetchdf()
conn.close()
pro_bar.progress(0.95, text="datos cargados")

df3.rename(columns={'FECHA': 'fecha'}, inplace=True)


# st.write(df4.dtypes, df1.dtypes, df2.dtypes, df3.dtypes)


df1['YEAR'] = df1['fecha'].dt.year  # maceda
df2['YEAR'] = df2['fecha'].dt.year  # conflictos
df3['YEAR'] = df3['fecha'].dt.year  # detenciones
# df4['YEAR'] = df4['emision'].dt.year  # licencias

# df4.rename(columns={'emision': 'fecha'}, inplace=True)
df1['fecha'] = pd.DatetimeIndex(df1['fecha'])
df2['fecha'] = pd.DatetimeIndex(df2['fecha'])
df3['fecha'] = pd.DatetimeIndex(df3['fecha'])
df4['fecha'] = pd.DatetimeIndex(df4['emision'])


df1_counts = df1.groupby(df1['fecha'].dt.to_period('M')
                         ).size().reset_index(name='count')
df2_counts = df2.groupby(df2['fecha'].dt.to_period('M')
                         ).size().reset_index(name='count')
df3_counts = df3.groupby(df3['fecha'].dt.to_period('M')
                         )['CASOS'].sum().reset_index(name='count')
df4_counts = df4.groupby(df4['fecha'].dt.to_period('M')
                         ).size().reset_index(name='count')

df1_counts['fuente'] = 'maceda'
df2_counts['fuente'] = 'conflictos'
df3_counts['fuente'] = 'detenciones'
df4_counts['fuente'] = 'licencias'

pro_bar.empty()
st.write(df1_counts.head(), df2_counts.head(),
         df3_counts.head(), df4_counts.head())


# st.write(plt.rcParams.keys())
# ---------------------------------------------------------
# GRÁFICO DE SERIE DE TIEMPO
# ---------------------------------------------------------
resultado = pd.concat(
    [df1_counts, df2_counts, df3_counts, df4_counts], ignore_index=True)
resultado = resultado[resultado['fecha'] > '2004-01-01']

st.write(resultado.head(100))
# resultado['fecha'] = resultado['fecha'].dt.to_period('M').dt.to_timestamp()

# Create the plot with updated date format
# fig = px.line(resultado, x='fecha', y='count', color='fuente',
#               title='Count over Time by Source')

# st.plotly_chart(fig, use_container_width=True)


# fig = px.line(resultado, x='fecha', y='count', color='fuente')
# fig.update_xaxes(ticks="outside", ticklen=10,
#                  showgrid=True, showline=True,
#                  minor=dict(ticks="inside", showgrid=True, ticklen=4))
# fig.update_yaxes(showgrid=True, showline=True)
# fig.update_layout(showlegend=True, xaxis_title="Fecha", yaxis_title="",
#                   title={'text': 'Conflictos en el tiempo',
#                          'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5,
#                          'font': {'size': 18}},
#                   height=600,
#                   xaxis_tickangle=-45,
#                   xaxis=dict(
#                       tickmode='linear',
#                       tickformat='%b-%Y',
#                       dtick='M7',
#                       ticklabelmode='period'
#                   ))
# st.plotly_chart(fig, use_container_width=True)

pro_bar.empty()
# ---------------------------------------------------------
# CORRELACIONES POR PERIODO
# ---------------------------------------------------------
# Agrupar y contar ocurrencias de conflictos en maceda_df
st.subheader('Correlaciones por períodos')
st.markdown("""Establecimiento de de posibles correlaciones entre los diferentes tipos de eventos/conflictos analizados por la cantidad de ocurrencia agrupadas por mes. 
            

No se detecta una correlación fuerte entre los eventos de maceda y los conflictos, detenciones o licencias médicas. Sin embargo, se observa una correlación positiva entre los conflictos y las detenciones, lo que sugiere que en las regiones con más conflictos también hay más detenciones. Por otro lado, la correlación negativa entre los conflictos y los eventos de maceda indica que estos eventos no suelen coincidir en las mismas regiones.      
""")

df1['año_mes'] = df1['fecha'].dt.to_period('M')
df2['año_mes'] = df2['fecha'].dt.to_period('M')
df3['año_mes'] = df3['fecha'].dt.to_period('M')
df4['año_mes'] = df4['fecha'].dt.to_period('M')

# Agrupar los datos de conflictos por mes a nivel nacional
maceda_nacional = df1.groupby(
    'año_mes').size().reset_index(name='maceda_count')
conflictos_nacional = df2.groupby(
    'año_mes').size().reset_index(name='conflictos_count')
detenciones_nacional = df3.groupby(
    'año_mes')['CASOS'].sum().reset_index(name='detenciones_count')
licencias_nacional = df4.groupby(
    'año_mes').size().reset_index(name='licencias_count')

# Combinar los datos agrupados por mes
combined_mc = pd.merge(
    maceda_nacional, conflictos_nacional, on='año_mes', how='inner')
combined_md = pd.merge(
    maceda_nacional, detenciones_nacional, on='año_mes', how='inner')
combined_ml = pd.merge(
    maceda_nacional, licencias_nacional, on='año_mes', how='inner')
# detenciones y licencias médicas
combined_dl = pd.merge(detenciones_nacional,
                       licencias_nacional, on='año_mes', how='inner')

corr_maceda_conflictos = combined_mc['maceda_count'].corr(
    combined_mc['conflictos_count'])
corr_maceda_detenciones = combined_md['maceda_count'].corr(
    combined_md['detenciones_count'])
corr_maceda_licencias = combined_ml['maceda_count'].corr(
    combined_ml['licencias_count'])
corr_detencion_licencias = combined_dl['detenciones_count'].corr(
    combined_dl['licencias_count'])

col1, col2 = st.columns(2, gap="small")
with col1:
    st.write(f'Correlación entre Maceda y Conflictos: {
        corr_maceda_conflictos}')

    # Plot correlation matrix
    sns.set(style="whitegrid")

    # Crear el gráfico de dispersión
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='maceda_count', y='conflictos_count',
                    data=combined_mc)

    # Añadir una línea de tendencia
    sns.regplot(x='maceda_count', y='conflictos_count',
                data=combined_mc, scatter=False, color='red')

    # Añadir títulos y etiquetas
    plt.title('Correlación entre Maceda y Conflictos por Mes', fontsize=16)
    plt.xlabel('Número de conflictos indígenas', fontsize=14)
    plt.ylabel('Número de conflictos', fontsize=14)
    # Show the plot
    st.pyplot(plt)

with col2:
    st.write(f'Correlación entre Maceda y Detenciones: {
        corr_maceda_detenciones}')

    # Crear el gráfico de dispersión
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='maceda_count', y='detenciones_count',
                    data=combined_md)

    sns.regplot(x='maceda_count', y='detenciones_count',
                data=combined_md, scatter=False, color='red')

    plt.title('Correlación entre Maceda y Detenciones por Mes', fontsize=16)
    plt.xlabel('Número de conflictos indígenas', fontsize=14)
    plt.ylabel('Número de detenciones', fontsize=14)
    # Show the plot
    st.pyplot(plt)

st.divider()

col3, col4 = st.columns(2, gap="small")
with col3:
    st.write(f'Correlación Maceda y Licencias: {
        corr_maceda_licencias}')

    # Crear el gráfico de dispersión
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='maceda_count', y='licencias_count',
                    data=combined_ml)

    sns.regplot(x='maceda_count', y='licencias_count',
                data=combined_ml, scatter=False, color='red')

    plt.title('Correlación entre Maceda y Licencias', fontsize=16)
    plt.xlabel('Número de conflictos indígenas', fontsize=14)
    plt.ylabel('Número de licencias médicas', fontsize=14)
    # Show the plot
    st.pyplot(plt)

with col4:
    st.write(f'Correlación Detenciones y Licencias: {
        corr_detencion_licencias}')
    # Crear el gráfico de dispersión
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='detenciones_count', y='licencias_count',
                    data=combined_dl)

    sns.regplot(x='detenciones_count', y='licencias_count',
                data=combined_dl, scatter=False, color='red')

    plt.title('Correlación entre Detencionesa y Licencias', fontsize=16)
    plt.xlabel('Número de detenciones', fontsize=14)
    plt.ylabel('Número de licencias médicas', fontsize=14)
    # Show the plot
    st.pyplot(plt)

st.markdown("""
    El gráfico de dispersión muestra la relación entre el número de detenciones por mes y el número de conflictos por mes en Chile. La línea de tendencia en rojo indica una ligera tendencia inversa entre los conflictos y las detenciones, consistente con la correlación negativa calculada anteriormente.

    Observaciones

    - Tendencia Inversa: La ligera tendencia descendente en la línea de tendencia sugiere que a medida que aumentan los conflictos, las detenciones tienden a disminuir ligeramente, y viceversa.
    - Distribución de Datos: La mayor parte de los datos están dispersos, indicando que la relación entre detenciones y conflictos no es muy fuerte.         
                """)

# ---------------------------------------------------------
# CORRELACIONES POR REGION
# ---------------------------------------------------------
st.subheader('Correlaciones a nivel regional')

# Agrupar los datos de conflictos por región y mes
maceda_region_mes = df1.groupby(
    ['año_mes', 'region']).size().reset_index(name='maceda_count')
conflictos_region_mes = df2.groupby(
    ['año_mes', 'region']).size().reset_index(name='conflictos_count')
detenciones_region_mes = df3.groupby(
    ['año_mes', 'region'])['CASOS'].sum().reset_index(name='detenciones_count')
# --------------------------
# correlaciones por region maceda y conflictos
combined_region_mes_mc = pd.merge(maceda_region_mes, conflictos_region_mes, on=[
    'año_mes', 'region'], how='inner')
r_corr_maceda_conflictos = combined_region_mes_mc['maceda_count'].corr(
    combined_region_mes_mc['conflictos_count'])
# -------------------------
# correlaciones por region maceda y detenciones
combined_region_mes_md = pd.merge(maceda_region_mes, detenciones_region_mes, on=[
    'año_mes', 'region'], how='inner')
r_corr_maceda_detenciones = combined_region_mes_md['maceda_count'].corr(
    combined_region_mes_md['detenciones_count'])
# -------------------------
# correlaciones por region conflictos y detenciones
combined_region_mes_cd = pd.merge(conflictos_region_mes, detenciones_region_mes, on=[
    'año_mes', 'region'], how='inner')
r_corr_conflictos_detenciones = combined_region_mes_cd['conflictos_count'].corr(
    combined_region_mes_cd['detenciones_count'])

coll, colr = st.columns(2, gap="small")
with coll:
    st.write(f'Maceda y Conflictos nivel regional: {
        r_corr_maceda_conflictos}')

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='maceda_count', y='conflictos_count',
                    data=combined_region_mes_mc)

    sns.regplot(x='maceda_count', y='conflictos_count',
                data=combined_region_mes_mc, scatter=False, color='red')

    plt.title('Maceda y conflictos a nivel regional', fontsize=16)
    plt.xlabel('Número de conflictos indígenas', fontsize=14)
    plt.ylabel('Número de conflictos', fontsize=14)
    # Show the plot
    st.pyplot(plt)

with colr:
    st.write(f'Correlación entre Maceda y Detenciones: {
        corr_maceda_detenciones}')

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='maceda_count', y='detenciones_count',
                    data=combined_region_mes_md)

    sns.regplot(x='maceda_count', y='detenciones_count',
                data=combined_region_mes_md, scatter=False, color='red')

    plt.title('Maceda y detenciones a nivel regional', fontsize=16)
    plt.xlabel('Número de conflictos indígenas', fontsize=14)
    plt.ylabel('Número de detenciones', fontsize=14)
    # Show the plot
    st.pyplot(plt)

col3, col4 = st.columns(2, gap="small")
with col3:
    st.write(f'Correlación entre Conflictos y Detenciones: {
        r_corr_conflictos_detenciones}')

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='conflictos_count', y='detenciones_count',
                    data=combined_region_mes_cd)

    sns.regplot(x='conflictos_count', y='detenciones_count',
                data=combined_region_mes_cd, scatter=False, color='red')

    plt.title('Conflictos y detenciones a nivel regional', fontsize=16)
    plt.xlabel('Número de conflictos', fontsize=14)
    plt.ylabel('Número de detenciones', fontsize=14)
    # Show the plot
    st.pyplot(plt)

with col4:
    st.empty()

# ---------------------------------------------------------
# CORRELACIONES A NIVEL PROVINCIA
# ---------------------------------------------------------

st.subheader('Correlaciones a nivel provincia')
df1['comuna'] = df1['comuna'].str.upper()
df1['provincia'] = df1['provincia'].str.upper()
df1['region'] = df1['region'].str.upper()

df2['comuna'] = df2['comuna'].str.upper()
df2['provincia'] = df2['provincia'].str.upper()
df2['region'] = df2['region'].str.upper()

df3['comuna'] = df3['comuna'].str.upper()
df3['provincia'] = df3['provincia'].str.upper()
df3['region'] = df3['region'].str.upper()

# Agrupar los datos de conflictos por provincia
maceda_provincia = df1.groupby(
    ['provincia']).size().reset_index(name='maceda')
conflictos_provincia = df2.groupby(
    ['provincia']).size().reset_index(name='conflictos')
detenciones_provincia = df3.groupby(
    ['provincia'])['CASOS'].sum().reset_index(name='detenciones')

# --------------------------
# correlaciones por region maceda y conflictos
combined_provincia_mc = pd.merge(maceda_provincia, conflictos_provincia, on=[
    'provincia'], how='inner')
c_corr_maceda_conflictos = combined_provincia_mc['maceda'].corr(
    combined_provincia_mc['conflictos'])

# --------------------------
# correlaciones por region maceda y detenciones
combined_provincia_md = pd.merge(maceda_provincia, detenciones_provincia, on=[
    'provincia'], how='inner')
c_corr_maceda_detenciones = combined_provincia_md['maceda'].corr(
    combined_provincia_md['detenciones'])

# --------------------------
# correlaciones por region conflictos y detenciones
combined_provincia_cd = pd.merge(conflictos_provincia, detenciones_provincia, on=[
    'provincia'], how='inner')
c_corr_conflictos_detenciones = combined_provincia_cd['conflictos'].corr(
    combined_provincia_cd['detenciones'])

colc1, colc2 = st.columns(2, gap="small")
with colc1:
    st.write(f'Maceda y Conflictos nivel: {
        c_corr_maceda_conflictos}')

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='maceda', y='conflictos',
                    data=combined_provincia_mc)

    sns.regplot(x='maceda', y='conflictos',
                data=combined_provincia_mc, scatter=False, color='red')

    plt.title('Maceda y conflictos', fontsize=16)
    plt.xlabel('Número de conflictos indígenas', fontsize=14)
    plt.ylabel('Número de conflictos', fontsize=14)
    st.pyplot(plt)

with colc2:
    st.write(f'Maceda y Detenciones : {
        c_corr_maceda_detenciones}')

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='maceda', y='detenciones',
                    data=combined_provincia_md)

    sns.regplot(x='maceda', y='detenciones',
                data=combined_provincia_md, scatter=False, color='red')

    plt.title('Maceda y detenciones', fontsize=16)
    plt.xlabel('Número de conflictos indígenas', fontsize=14)
    plt.ylabel('Número de detenciones', fontsize=14)
    st.pyplot(plt)

colc3, colc4 = st.columns(2, gap="small")
with colc3:
    st.write(f'Conflictos y Detenciones : {
        c_corr_conflictos_detenciones}')

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='conflictos', y='detenciones',
                    data=combined_provincia_cd)

    sns.regplot(x='conflictos', y='detenciones',
                data=combined_provincia_cd, scatter=False, color='red')

    plt.title('Conflictos y detenciones', fontsize=16)
    plt.xlabel('Número de conflictos sociales', fontsize=14)
    plt.ylabel('Número de detenciones', fontsize=14)
    st.pyplot(plt)

with colc4:
    st.empty()

# st.write(combined_provincia_mc, combined_provincia_md, combined_provincia_cd)
st.markdown(f"""
##### Resultados de las Correlaciones Espaciales
No se incluyen las licencias médicas en este análisis debido a que aporta información de locación (comuna o región). Las correlaciones espaciales entre los diferentes tipos de eventos son las siguientes:

- Detenciones y Conflictos: {c_corr_conflictos_detenciones} (alta y positiva)
- Detenciones y Maceda: {c_corr_maceda_detenciones} (baja y negativa)
- Conflictos y Maceda: {c_corr_maceda_conflictos} (baja y negativa)
""")
st.markdown("""
**Interpretación**


**Detenciones y Conflictos**: Existe una alta correlación positiva entre las detenciones y los conflictos en las mismas regiones. Esto sugiere que en las regiones donde hay más conflictos, también hay más detenciones.



**Detenciones y Maceda**: La correlación negativa indica que en las regiones con más eventos de detenciones, hay menos eventos de maceda, aunque esta relación es baja.



**Conflictos y Maceda**: De manera similar, hay una correlación negativa entre los conflictos y los eventos de maceda, indicando que estas ocurrencias no suelen coincidir en las mismas regiones.



Estos resultados sugieren que hay una relación espacial significativa entre las detenciones y los conflictos, pero no tanto con los eventos de maceda.""")
