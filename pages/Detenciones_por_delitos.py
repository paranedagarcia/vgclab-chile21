from funciones import load_data_csv
import duckdb
from funciones import menu_pages, convert_df
from datetime import datetime
from pygwalker.api.streamlit import StreamlitRenderer
from millify import millify
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from io import BytesIO
import base64
from streamlit_extras.metric_cards import style_metric_cards
import streamlit.components.v1 as components
import streamlit as st
import time
import numpy as np
import pandas as pd
import sys
import os


# librerias


# import seaborn as sns


# configuration
st.set_page_config(
    page_title="Eficiencia de justicia",
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
progress_text = "Cargando datos..."
pro_bar = st.progress(0.15, text=progress_text)

# comunas_file = "https://data.vgclab.cl/public_data/comunas.csv"
# comunas = pd.read_csv(comunas_file, header=0, sep=";")
# regiones = comunas["region"].unique().tolist()
# provincias = comunas["provincia"].unique().tolist()

# public = "https://vgclabfiles.blob.core.windows.net/public/"
# filename = public+"dataset_detenciones.csv"
# reader = pd.read_csv(filename, chunksize=10000, nrows=10000)
# df = pd.concat([x for x in reader], ignore_index=True)
try:
    conn = duckdb.connect('vglab.db')
    query = "SELECT * FROM detenciones"
    df = conn.execute(query).fetchdf()
    conn.close()
except:
    st.error("Error al cargar los datos")
    st.stop()

pro_bar.progress(0.3, text="Estableciendo métricas...")

df['comuna'] = df['comuna'].str.upper()
df['provincia'] = df['provincia'].str.upper()
df['region'] = df['region'].str.upper()

df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y-%m-%d')

# Extract year and create 'YEAR' column
df['YEAR'] = df['FECHA'].dt.year

# meses
month_names = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}
month_shorts = {
    1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr',
    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago',
    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
}


# csv = convert_df(df)
with st.sidebar:
    delitos = df["DELITO"].unique().tolist()
    delitos = [x for x in delitos if str(x) != 'nan']
    delito_selection = st.sidebar.multiselect("Seleccione delitos a analizar:", delitos, [
        'Rb_Violencia_o_Intimidación', 'Rb_Sorpresa', 'Rb_Fuerza', 'Rb_Vehìculo', 'Hurtos', 'Homicidio', 'Lesiones', 'Violación'])
    if delito_selection is not None:
        df = df[df["DELITO"].isin(delito_selection)]
    else:
        df = df


# filter delitos
# selected_unidad = st.sidebar.selectbox(
#     "Seleccione agrupación", ["Ninguna", "Comuna", "Provincia", "Región"])

st.subheader("Eficiencia de la Justicia")
# st.subheader("Detenciones " + selected_unidad)

# --------------------------
# METRICAS
# --------------------------
delito_mayor = df.groupby('DELITO')['CASOS'].sum().nlargest(
    1).reset_index(name='max')

# st.write("Total de detenciones: ", df.shape[0])
detenciones_totales = df.shape[0]
delitos_totales = df["DELITO"].nunique()

# Calcula el total de casos para cada tipo de delito
delitos_suma = df.groupby('DELITO')['CASOS'].sum().reset_index()

# Calcula el total de casos para el cálculo del porcentaje
total_cases = delitos_suma['CASOS'].sum()


# Ordena los delitos por número de casos para obtener los delitos más comunes
# crime_counts_sorted = crime_counts.sort_values('CASOS', ascending=False)

tabPanel, tabTable, tabBleau = st.tabs(
    ["Panel", "Tabla", "Informes"])

with tabPanel:
    # st.write(delito_mayor.iloc[0, 0] + " es el delito más común con " +
    #         str(delito_mayor.iloc[0, 1]) + " casos.")
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    col1.metric("Total de detenciones", millify(
        detenciones_totales, precision=2))
    col2.metric("Tipo de delitos", delitos_totales)
    col3.metric(delito_mayor.iloc[0, 0], millify(
        delito_mayor.iloc[0, 1], precision=2))
    col4.metric("Comuna", "La Florida")

    style_metric_cards()

    pro_bar.progress(0.5, text="Construyendo gráficos...")

    det = df.groupby(["comuna", "DELITO", "YEAR"]).agg({'CASOS': sum})

    dfcomuna = df.groupby('comuna')['CASOS'].sum().nlargest(
        10).reset_index(name='suma')
    dfcomuna = dfcomuna.sort_values(by='suma', ascending=False)

    dfregion = df.groupby('region')['CASOS'].sum().reset_index(name='suma')
    dfregion = dfregion.sort_values(by='suma', ascending=False)

    # ------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # TENDENCIA DE DELITOS VIOLENTOS
    # ------------------------------------------------------------
    st.divider()

    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df['YEAR_MONTH'] = df['FECHA'].dt.strftime('%Y-%m')

    # anño mes
    delitos_por_fecha = df.groupby('YEAR_MONTH')['CASOS'].sum().reset_index()
    # delitos_por_fecha['YEAR_MONTH'] = delitos_por_fecha['YEAR_MONTH'].astype(
    #    str)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=delitos_por_fecha['YEAR_MONTH'], y=delitos_por_fecha['CASOS'], mode='lines+markers',
        name='CASOS'))

    fig.update_layout(title={'text': 'Tendencia de Delitos Violentos en el Tiempo en Chile',
                             'xanchor': 'center',
                             'yanchor': 'top', 'y': 0.95, 'x': 0.5, 'font': {'size': 18}},
                      xaxis_title='Año y Mes',
                      yaxis_title='Casos',
                      xaxis_tickangle=-45,
                      xaxis=dict(
                          tickmode='linear',
                          tickformat='%b-%Y',
                          dtick='M5',
                          ticklabelmode='period'
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Group by year, month, and crime type, summing the cases
    monthly_crime_comparison = df.groupby(['YEAR', 'mes', 'DELITO'])[
        'CASOS'].sum().reset_index()

    # Create a new column for 'year-month' to use as x-axis
    monthly_crime_comparison['Año-Mes'] = monthly_crime_comparison['YEAR'].astype(
        str) + '-' + monthly_crime_comparison['mes'].astype(str).str.zfill(2)

    # Define the month names in Spanish
    month_names_spanish = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }

    # Map month numbers to month names
    monthly_crime_comparison['Mes'] = monthly_crime_comparison['mes'].map(
        month_names_spanish)

    # Create a line chart to compare monthly crimes across different years and crime types
    fig_monthly_comparison = px.line(
        monthly_crime_comparison,
        x='Año-Mes',
        y='CASOS',
        color='DELITO',
        title='Comparación Mensual de Delitos por Año y Tipo de Delito',
        labels={'Año-Mes': 'Fecha', 'CASOS': 'Casos',
                'DELITO': 'Tipo de Delito'},
        color_discrete_sequence=px.colors.qualitative.Prism
    )

    # Update the layout to show dates in Spanish
    fig_monthly_comparison.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Casos',
        legend_title_text='Tipo de Delito',
        xaxis_tickangle=-45,
        xaxis=dict(
                          tickmode='linear',
                          tickformat='%b-%Y',
                          dtick='M5',
                          ticklabelmode='period'
        ),
        title={'xanchor': 'center',
               'yanchor': 'top', 'y': 0.95, 'x': 0.5, 'font': {'size': 18}}
    )
    st.plotly_chart(fig_monthly_comparison, use_container_width=True)

    st.markdown("""Distribución de los delitos por regiones o comunas, resaltando las áreas con mayor incidencia de delitos.
                
El gráfico muestra la distribución de delitos por región. 
    
Las regiones están ordenadas de mayor a menor número de casos, permitiendo identificar fácilmente cuáles son las regiones con más incidencias de delitos.""")

    # ------------------------------------------------------------
    # DISTRIBUCION GEOGRAFICA DE DELITOS POR COMUNA
    # ------------------------------------------------------------
    st.divider()
    # ------------------------------------------------

    delitos_por_comuna = df.groupby(
        'comuna')['CASOS'].sum().reset_index()
    delitos_por_comuna.sort_values('CASOS', ascending=False, inplace=True)
    top_comunas = delitos_por_comuna.head(25)

    fig = px.bar(top_comunas, x='comuna', y='CASOS',
                 title='Top 25 Comunas con Mayor Incidencia de Delitos',
                 labels={'comuna': 'comuna', 'CASOS': 'Número de Casos'},
                 color='CASOS',  # Usar el número de casos para el color de las barras
                 color_discrete_sequence=px.colors.qualitative.Prism)
    fig.update_layout(xaxis_title='comuna',
                      yaxis_title='Número de Casos',
                      xaxis_tickangle=-45,
                      title={'xanchor': 'center',
                             'yanchor': 'top', 'y': 0.95, 'x': 0.5, 'font': {'size': 18}}
                      )
    st.plotly_chart(fig, use_container_width=True)

    st.write("Distribución comunal de delitos: distribución de los delitos por 'comuna' resaltando las comunas con mayor incidencia de delitos.")

    # ------------------------------------------------------------
    # DISTRIBUCION GEOGRAFICA DE DELITOS POR COMUNA POR DELITO
    # ------------------------------------------------------------
    st.divider()

    delitos_comuna_tipo = df.groupby(['comuna', 'DELITO'])[
        'CASOS'].sum().reset_index()
    top_comunas_list = delitos_por_comuna.head(10)['comuna'].tolist()
    delitos_comuna_tipo_top = delitos_comuna_tipo[delitos_comuna_tipo['comuna'].isin(
        top_comunas_list)]

    fig = px.bar(delitos_comuna_tipo_top, x='comuna', y='CASOS', color='DELITO',
                 title='Distribución de Delitos por Comuna, Diferenciando por Tipo de Delito',
                 labels={'comuna': 'Comuna', 'CASOS': 'Número de Casos',
                         'DELITO': 'Tipo de Delito'},
                 color_discrete_sequence=px.colors.qualitative.Prism)
    fig.update_layout(xaxis_title='Comuna',
                      yaxis_title='Número de Casos',
                      legend_title='Tipo de Delito',
                      xaxis_tickangle=-45,
                      title={'xanchor': 'center',
                             'yanchor': 'top', 'y': 0.95, 'x': 0.5, 'font': {'size': 18}},
                      yaxis=dict(
                          tickfont=dict(size=14)),
                      xaxis=dict(
                          tickfont=dict(size=14)),
                      height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.write("Distribución comunal de delitos: distribución de los delitos por 'comuna' diferenciando los tipos de delitos, resaltando las comunas con mayor incidencia de delitos.")
    st.write("Este gráfico no solo destaca las comunas con la mayor incidencia total de delitos, sino que también muestra la proporción de diferentes tipos de delitos dentro de cada comuna, lo que proporciona una comprensión más profunda de la naturaleza de los delitos en estas áreas.")

    # ------------------------------------------------------------
    # TENDENCIA POR TEMPORADA
    # ------------------------------------------------------------

    # Define seasons based on the given months

    def determine_season(month):
        if month in [12, 1, 2]:
            return 'Verano'
        elif month in [3, 4, 5]:
            return 'Otoño'
        elif month in [6, 7, 8]:
            return 'Invierno'
        elif month in [9, 10, 11]:
            return 'Primavera'

    # Apply the function to determine the season
    df['Temporada'] = df['mes'].apply(determine_season)

    # Group by season and crime type, summing the cases
    seasonal_crime_distribution = df.groupby(['Temporada', 'DELITO'])[
        'CASOS'].sum().reset_index()

    # Create a bar chart for seasonal crime distribution
    fig_seasonal_crime = px.bar(
        seasonal_crime_distribution,
        x='Temporada',
        y='CASOS',
        color='DELITO',
        title='Distribución de Delitos por Temporadas',
        labels={'Temporada': 'Temporada',
                'CASOS': 'Casos', 'DELITO': 'Tipo de Delito'},
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_seasonal_crime.update_layout(title={'xanchor': 'center',
                                            'yanchor': 'top', 'y': 0.95, 'x': 0.5, 'font': {'size': 18}})
    st.plotly_chart(fig_seasonal_crime, use_container_width=True)

    # ------------------------------------------------------------
    # TENDENCIA DE DELITOS POR DIA DE LA SEMANA
    # ------------------------------------------------------------

    df['dow'] = df['FECHA'].dt.day_name(locale='es_ES')

    # Group by day of the week and sum the cases
    day_of_week_trends = df.groupby('dow')['CASOS'].sum().reset_index()

    # Ensure the days of the week are ordered correctly
    dow_order = ['lunes', 'martes', 'miércoles',
                 'jueves', 'viernes', 'sábado', 'domingo']
    day_of_week_trends['day_of_week'] = pd.Categorical(
        day_of_week_trends['dow'], categories=dow_order, ordered=True)
    day_of_week_trends = day_of_week_trends.sort_values('dow')

    # Create a bar chart for cases by day of the week
    fig_day_of_week = px.bar(
        day_of_week_trends,
        x='dow',
        y='CASOS',
        title='Tendencia de delitos por día de la semana',
    )
    fig_day_of_week.update_layout(title={'xanchor': 'center',
                                         'yanchor': 'top', 'y': 0.95, 'x': 0.5, 'font': {'size': 18}})
    st.plotly_chart(fig_day_of_week, use_container_width=True)

    # ------------------------------------------------------------
    # COMPARACION ANUAL
    # ------------------------------------------------------------
    # Year-over-Year Comparison of Crime Cases
    yearly_comparison = df.groupby(['YEAR'])['CASOS'].sum().reset_index()

    fig_yearly_comparison = px.bar(
        yearly_comparison,
        x='YEAR',
        y='CASOS',
        title='Comparación Anual de Total de Casos',
        labels={'YEAR': 'Año', 'CASOS': 'Casos'}
    )
    fig_yearly_comparison.update_layout(
        xaxis_tickangle=0,
        xaxis=dict(
            tickmode='linear',
            tickformat='%Y',
            dtick='Y1',
            ticklabelmode='period'
        ),
        title={'xanchor': 'center',
               'yanchor': 'top', 'y': 0.95, 'x': 0.5, 'font': {'size': 18}},
    )
    st.plotly_chart(fig_yearly_comparison, use_container_width=True)
# ------------------------------------------------------------
# TABLA DE DATOS
# ------------------------------------------------------------

with tabTable:
    dftabla = df.copy()
    dftabla.rename(columns={'FECHA': 'fecha'}, inplace=True)
    dftabla['fecha'] = dftabla['fecha'].dt.date
    st.dataframe(dftabla, height=600)
    # with st.sidebar:
    #     st.download_button("Descargar datos (csv)",
    #                        convert_df(dftabla), "detenciones.csv",
    #                        "text/csv", key="donwload-csv")

# --------------------------
with tabBleau:  # graficos personalizados
    # @st.cache_resource
    def get_pyg_renderer() -> "StreamlitRenderer":
        return StreamlitRenderer(df, spec_io_mode="rw")

    renderer = get_pyg_renderer()
    renderer.explorer()


pro_bar.empty()
# with tabBleau:
#     report = pgw.walk(df, return_html=True)
#     components.html(report, height=1000, scrolling=True)

# ------------------------------------------------------------
# TAB INFO
# ------------------------------------------------------------
# with tabInfo:
#     st.write("Información")

#     st.write("Detenciones en Chile")
#     st.write("Fecha de actualización: ",
#              datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#     st.write("Autor: Patricio Araneda")
#     st.write("Fuente: [Fundación Chile 21](https://chile21.cl/)")
#     st.write("San Sebastián 2807, Las Condes, Santiago de Chile")
