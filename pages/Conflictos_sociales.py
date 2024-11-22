'''
2024-20-10
cambio de color de gráficos a modelo de color prism
origen de datos desde database duckdb

'''
# librerias
import os
import pandas as pd
import numpy as np
import time
import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.metric_cards import style_metric_cards

import base64
from io import BytesIO
from PIL import Image

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
# import mpld3
# from mpld3 import plugins
from millify import millify
from pygwalker.api.streamlit import StreamlitRenderer

from datetime import datetime
from funciones import menu_pages, convert_df
import duckdb

# configuration
st.set_page_config(
    page_title="Conflictos",
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


def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)

    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))


def df_filterfecha(message, df):
    dates_selection = st.sidebar.slider('%s' % (message),
                                        min_value=min(df['fecha']),
                                        max_value=max(df['fecha']),
                                        value=(min(df['fecha']), max(df['fecha'])))
    mask = df['fecha'].between(*dates_selection)
    number_of_result = df[mask].shape[0]
    filtered_df = df[mask]
    return filtered_df


progress_text = "Cargando datos..."
pro_bar = st.progress(0.15, text=progress_text)

menu_pages()

# carga de datos
public = "https://vgclabfiles.blob.core.windows.net/public/"
# public = "data/"
filename = public + "dataset_conflictos.csv"
# pandas large files
# reader = pd.read_csv(filename, chunksize=10000, nrows=10000, sep=";")
# df = pd.concat([x for x in reader], ignore_index=True)

try:
    conn = duckdb.connect('vglab.db')
    query = "SELECT * FROM conflictos"
    df = conn.execute(query).fetchdf()
    conn.close()
except:
    st.error("Error al cargar los datos")
    st.stop()

df['comuna'] = df['comuna'].str.upper()
df['provincia'] = df['provincia'].str.upper()
df['region'] = df['region'].str.upper()
df['fecha'] = pd.to_datetime(df[['año', 'mes', 'dia']].astype(str).agg(
    '-'.join, axis=1), errors='coerce', format='mixed', dayfirst=True)
# cambiar fecha al inicio
fecha = df.pop("fecha")
df.insert(1, "fecha", fecha)

pro_bar.progress(0.3, text="Estableciendo métricas...")

# metricas
# years = df["fecha"].dt.year.unique().tolist()
# years.insert(0, "Todos")


# medios
medios = df["medio"].unique().tolist()
medios.insert(0, "Todos")

# sidebar
#
# anual = st.sidebar.selectbox("Seleccione un año", years)

# if anual is not 'Todos':
#    df = df[df["fecha"].dt.year == anual]
# else:
#    pass

medio = st.sidebar.selectbox(
    "Medios", medios)
if medio != 'Todos':
    df = df[df["medio"] == medio]
else:
    df = df


# TIPO DE CONFLICTO SOCIAL
tipos = sorted(df["Tipo de conflicto social"].dropna().unique().tolist())

with st.sidebar.expander("Tipos de conflicto"):
    tipoconflicto = st.multiselect(
        "Conflicto", tipos, default=tipos)
    if tipoconflicto != None:
        df = df[df["Tipo de conflicto social"].isin(tipoconflicto)]
    else:
        df = df


heridos_manifestantes = df["Manifestantes heridos"].sum()
heridos_carabineros = df["Carabineros heridos"].sum()
muertos_manifestantes = df["Manifestantes muertos"].sum()
muertos_carabineros = df["Carabineros muertos"].sum()
arrestos = df["Arrestos"].sum()
heridos_personas = df["Personas heridas"].sum()
muertos_personas = df["Personas muertas"].sum()

# main
st.subheader("Evolución de conflictos en Chile (2008 - 2020)")

tabPanel, tabTable, tabBleau = st.tabs(
    ["Panel", "Tabla", "Informes"])

# --------------------------
with tabPanel:
    # --------------------------
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    col1.metric("Menciones", millify(len(df), precision=2))
    col2.metric("Manifestantes heridos", millify(
        heridos_manifestantes, precision=2))
    col3.metric("Carabineros heridos", millify(
        heridos_carabineros, precision=2))
    col4.metric("Arrestos", millify(arrestos, precision=2))

    col7, col8, col9, col10 = st.columns(4, gap="medium")
    col7.metric("Manifestantes muertos", muertos_manifestantes)
    col8.metric("Carabineros muertos", muertos_carabineros)
    col9.metric("Personas heridas", heridos_personas)
    col10.metric("Personas muertas", muertos_personas)

    style_metric_cards()

    pro_bar.progress(0.5, text="Construyendo gráficos...")

    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

    # ------------------------------------------------------------
    # MENCIONES EN MEDIOS DE COMUNICACION
    # ------------------------------------------------------------
    st.divider()

    # st.write("Menciones en Medios de Comunicación")
    # mencioines ne medios de comunicacion
    df_medios = df.groupby('medio').size().reset_index(name='cuenta')
    fig = px.bar(
        df_medios,
        x="medio",
        y="cuenta",
        title="Menciones en medios de comunicación",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_traces(hovertemplate='%{x}<br>Menciones: %{y}')
    fig.update_xaxes(showgrid=True, showline=True)
    fig.update_yaxes(showgrid=True, showline=True)
    fig.update_layout(showlegend=False,
                      xaxis_title="",
                      yaxis_title="",
                      # yaxis_tickformat="20,.2f"
                      yaxis=dict(tickformat=",.2r",), height=600
                      )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------
    # MANIFESTANTES
    # ------------------------------------------------------------
    st.divider()
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        fig = px.pie(
            names=["Manifestantes heridos",
                   "Carabineros heridos", "Personas heridas"],
            values=[heridos_manifestantes,
                    heridos_carabineros, heridos_personas],
            title="Manifestantes heridos vs Carabineros heridos vs Personas heridas",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig.update_traces(hovertemplate='%{label}<br> %{value}')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            names=["Manifestantes muertos", "Carabineros muertos"],
            values=[muertos_manifestantes, muertos_carabineros],
            title="Manifestantes muertos vs Carabineros muertos",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig.update_traces(hovertemplate='%{label}<br> %{value}')
        st.plotly_chart(fig, use_container_width=True)

    pro_bar.progress(0.7, text="Construyendo gráficos...")

    # ------------------------------------------------------------
    # ACTORES DEMANDANTES Y AFECTADOS
    # ------------------------------------------------------------

    df_demandante = df.groupby(
        'Actor demandante').size().reset_index(name='cuenta')

    with st.expander("Actores demandantes"):
        # ACTORES DEMANDANTES
        # actores_demandantes = df["Actor demandante"].unique().tolist()
        actores_demandantes = sorted(df["Actor demandante"].dropna(
        ).unique().tolist())

        actores_demandantes.insert(0, "Todos")

        actor_demandante = st.multiselect(
            "", actores_demandantes, default=actores_demandantes)
        if actor_demandante != "Todos":
            df_demandante = df_demandante[df_demandante["Actor demandante"].isin(
                actor_demandante)]
        else:
            df_demandante = df_demandante

    fig = px.bar(
        df_demandante,
        y="Actor demandante",
        x="cuenta",
        color="Actor demandante",
        title="Actores demandantes en conflictos sociales",
        orientation="h", height=700,
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_xaxes(showgrid=True, showline=True)
    fig.update_yaxes(showgrid=True, showline=True)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="",
                      title={
                          'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------------------------
    # ACTOR AFECTADO
    df_afectado = df.groupby(
        'Actor destino de protesta').size().reset_index(name='cuenta')

    with st.expander("Actores afectados"):
        # ACTORES AFECTADOS
        actores_afectados = sorted(
            df["Actor destino de protesta"].dropna().unique().tolist())
        actores_afectados.insert(0, "Todos")

        actor_afectado = st.multiselect(
            "", actores_afectados, default=actores_afectados)
        if actor_afectado != "Todos":
            df_afectado = df_afectado[df_afectado["Actor destino de protesta"].isin(
                actor_afectado)]
        else:
            df_afectado = df_afectado

    fig = px.bar(
        df_afectado,
        y="Actor destino de protesta",
        x="cuenta",
        color="Actor destino de protesta",
        title="Actores afectados por conflictos sociales",
        orientation="h", height=700,
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_xaxes(showgrid=True, showline=True)
    fig.update_yaxes(showgrid=True, showline=True)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="",
                      title={
                          'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
    st.plotly_chart(fig, use_container_width=True)

    pro_bar.progress(0.8, text="Construyendo gráficos...")

    # ------------------------------------------------------------
    # Tipos de conflictos por semana mes
    # ------------------------------------------------------------
    st.divider()

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        # ------------------------------------------------------------
        # DIA DE LA SEMANA
        # ------------------------------------------------------------

        # Convertir la columna 'fecha' correctamente para evitar el error
        # Crear una columna 'fecha' a partir de las columnas 'año', 'mes' y 'dia'
        df['fecha'] = pd.to_datetime(df[['año', 'mes', 'dia']].astype(
            str).agg('-'.join, axis=1), errors='coerce')

        # Extraer el día de la semana de la columna 'fecha'
        df['día_semana'] = df['fecha'].dt.day_name()

        # Agrupar los datos por día de la semana y contar las ocurrencias de conflictos
        conflicts_by_day = df['día_semana'].value_counts().reset_index()
        conflicts_by_day.columns = [
            'Día de la semana', 'Cantidad de Conflictos']

        # Ordenar los días de la semana
        ordered_days = ['Monday', 'Tuesday', 'Wednesday',
                        'Thursday', 'Friday', 'Saturday', 'Sunday']
        conflicts_by_day['Día de la semana'] = pd.Categorical(
            conflicts_by_day['Día de la semana'], categories=ordered_days, ordered=True)
        conflicts_by_day = conflicts_by_day.sort_values('Día de la semana')

        # Crear gráfico de barras para visualizar los conflictos por día de la semana
        fig = px.bar(conflicts_by_day, x='Día de la semana', y='Cantidad de Conflictos',
                     title='Distribución de Conflictos por Día de la Semana',
                     labels={'Cantidad de Conflictos': 'Cantidad de Conflictos',
                             'Día de la semana': 'Día de la Semana'},
                     template='plotly_white', height=600,
                     color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_layout(title={
                          'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ------------------------------------------------------------
        # MES
        # ------------------------------------------------------------

        # Convertir la columna 'fecha' correctamente para evitar el error
        # Crear una columna 'fecha' a partir de las columnas 'año', 'mes' y 'dia'
        # Mapeo de números de meses a nombres en español
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

        conflicts_by_month = df.groupby(
            'mes').size().reset_index(name='counts')
        # Mapear los nombres de los meses al dataframe
        conflicts_by_month['mes_nombre'] = conflicts_by_month['mes'].map(
            month_names)

        # Crear un gráfico de barras interactivo para mostrar los conflictos por mes del año con nombres en español
        fig = px.bar(conflicts_by_month, x='mes_nombre', y='counts',
                     title='Distribución de Conflictos por Mes del Año',
                     labels={'counts': 'Cantidad de Conflictos',
                             'mes_nombre': 'Mes'},
                     template='plotly_white', height=600,
                     color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_layout(title={
                          'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------
    # 1. Distribución anual de los conflictos
    # ------------------------------------------------------------
    st.divider()

    # Agrupar los datos por año y contar las ocurrencias de conflictos
    conflicts_by_year = df.groupby(
        'año').size().reset_index(name='counts')

    # Crear un gráfico de líneas interactivo para mostrar los conflictos a través de los años
    fig = px.line(conflicts_by_year, x='año', y='counts',
                  labels={'counts': 'Cantidad de Conflictos', 'año': 'Año'},
                  markers=True,
                  template='plotly_white')
    fig.update_layout(title={'text': 'Cantidad de Conflictos a Través de los Años',
                      'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
Se observa una variabilidad en el número de conflictos sociales a lo largo de los años.
Es relevante analizar si ciertos eventos históricos o cambios políticos coinciden con picos en la cantidad de conflictos.""")

    # ------------------------------------------------------------
    # 1. Distribución geográfica
    # ------------------------------------------------------------
    st.divider()
    st.markdown("<h4><center>Distribución geográfica</center></h4>",
                unsafe_allow_html=True)

    region_data = df['region'].value_counts().reset_index().head(20)
    region_data.columns = ['region', 'conteo']

    # Crear un gráfico de barras interactivo para visualizar la distribución de los conflictos sociales por comuna
    fig = px.bar(region_data, x='region', y='conteo',
                 labels={'region': 'Región',
                         'conteo': 'Cantidad de Conflictos'},
                 template='plotly_white', height=700,
                 color_discrete_sequence=px.colors.qualitative.Prism)
    fig.update_layout(title={'text': 'Distribución de los Conflictos por Región',
                      'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
Las regiones con mayor número de conflictos son Valparaíso y la Metropolitana.
Esto puede correlacionarse con la densidad poblacional y la actividad económica en estas regiones.

**Tipos de Conflictos:**

Se destacan ciertos tipos de noticias, lo que puede indicar las formas predominantes de manifestación o cobertura mediática.
Profundizar en los tipos más frecuentes podría revelar patrones específicos de descontento social.

**Cantidad de Participantes:**

La mayoría de los conflictos involucran un número relativamente bajo de participantes.
Sin embargo, hay eventos que movilizan a un gran número de personas, lo que podría estar relacionado con la importancia del motivo de la protesta.""")

    # ------------------------------------------------------------
    # distribución de los tipos de conflictos sociales por región
    # ------------------------------------------------------------
    st.divider()
    # Agrupar los datos por región y tipo de conflicto social, y contar las ocurrencias
    conflict_types_by_region = df.groupby(
        ['region', 'Tipo de conflicto social']).size().reset_index(name='counts')

    # Crear un gráfico de barras apiladas interactivo
    fig = px.bar(conflict_types_by_region, x='region', y='counts', color='Tipo de conflicto social',
                 labels={'counts': 'Cantidad',
                         'Tipo de conflicto social': 'Tipo de Conflicto'},
                 template='plotly_white', height=700,
                 color_discrete_sequence=px.colors.qualitative.Prism)
    fig.update_layout(title={'text': 'Distribución de los Tipos de Conflictos Sociales por Región',
                      'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        **Distribución de los Tipos de Conflictos Sociales por Región**

El gráfico de barras apiladas muestra la distribución de los diferentes tipos de conflictos sociales en cada región. Este análisis permite observar las particularidades de cada región respecto a la naturaleza de los conflictos reportados, proporcionando una visión más detallada de las dinámicas sociales en diferentes partes del país.
        """)

    # ------------------------------------------------------------
    # Distribución de los conflictos por comuna
    # ------------------------------------------------------------
    st.divider()
    comuna_data = df['comuna'].value_counts().reset_index().head(20)
    comuna_data.columns = ['comuna', 'conteo']

    # Crear un gráfico de barras interactivo para visualizar la distribución de los conflictos sociales por comuna
    fig = px.bar(comuna_data, x='comuna', y='conteo',
                 title='Distribución de los Conflictos Sociales por Comuna',
                 labels={'comuna': 'Comuna',
                         'conteo': 'Cantidad de Conflictos'},
                 template='plotly_white',
                 color_discrete_sequence=px.colors.qualitative.Prism)
    fig.update_layout(title={'text': 'Distribución de los Conflictos Sociales por Comuna',
                      'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Distribución de los Conflictos Sociales por Comuna**

El gráfico muestra las 20 comunas con el mayor número de conflictos sociales. Este análisis puede ayudar a identificar las áreas más conflictivas dentro de las regiones y provincias.
""")
    # ------------------------------------------------------------
    # 4. Cantidad de participantes
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Conflictos por sector y región
    # ------------------------------------------------------------
    st.divider()

    # Crear un gráfico de barras interactivo para mostrar la distribución de los conflictos sociales por sector económico y región
    # Agrupar los datos por región y sector económico (usando 'Grupo social' como proxy para el sector económico)
    sector_region_data = df.groupby(
        ['region', 'sector demandante']).size().reset_index(name='count')

    # Crear un gráfico de barras apilado interactivo para mostrar la distribución de conflictos sociales
    fig = px.bar(sector_region_data, x='region', y='count', color='sector demandante',
                 labels={'count': 'Cantidad', 'region': 'Región',
                         'Actor demandante': 'Sector Demandante'},
                 template='plotly_white', height=700,
                 color_discrete_sequence=px.colors.qualitative.Prism)
    fig.update_layout(
        title={'text': 'Distribución de Conflictos Sociales por Sector Demandante y Región', 'xanchor': 'center',
               'yanchor': 'top', 'y': .95, 'x': 0.5,
               'font': {'size': 18}})
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
El gráfico de barras apiladas muestra cómo se distribuyen los conflictos sociales según el sector económico en cada región.

<p><strong>Diversidad regional:</strong></p> Se puede observar la variación en los sectores económicos involucrados en conflictos según la región.

<p><strong>Sectores predominantes por región:</strong></p> 
Algunas regiones tienen conflictos concentrados en ciertos sectores económicos específicos, lo que puede reflejar las características económicas y sociales de esas áreas.
Este análisis proporciona una visión más detallada de cómo los conflictos se distribuyen geográficamente y cuáles sectores son más afectados en diferentes regiones.""", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # TACTICAS DE PROTESTA
    # ------------------------------------------------------------
#     st.divider()
#     tactics_columns_available = [
#         'Tácticas pacificas 1', 'Tácticas pacificas 2', 'Tácticas artísticas',
#         'Tácticas disruptivas', 'Tácticas autodestructivas', 'Tácticas violentas'
#     ]

#     # Contar la frecuencia de cada táctica
#     tactics_data_available = df[tactics_columns_available].apply(
#         pd.Series.value_counts).fillna(0)

#     # Sumar las frecuencias para cada táctica
#     tactics_counts_available = tactics_data_available.sum().reset_index()
#     tactics_counts_available.columns = ['Táctica', 'Frecuencia']

#     # Crear un gráfico de barras interactivo
#     fig = px.bar(tactics_counts_available, x='Táctica', y='Frecuencia',
#                  labels={'Frecuencia': 'Número de conflictos',
#                          'Táctica': 'Táctica'},
#                  template='plotly_white', height=600)
#     fig.update_layout(showlegend=False,
#                       title={'text': 'Frecuencia de las Tácticas Usadas en los Conflictos Sociales', 'xanchor': 'center',
#                              'yanchor': 'top', 'y': 0.95, 'x': 0.5,
#                                                'font': {'size': 18}})
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown("""
#                 ##### Análisis de Tácticas Usadas en Conflictos**

# El gráfico de barras horizontales muestra la frecuencia con la que se utilizan diferentes tácticas en los conflictos sociales.

# **Tácticas pacíficas:**

# Son las más frecuentemente reportadas, lo que puede indicar una tendencia general hacia formas no violentas de protesta.

# **Tácticas artísticas:**

# También son utilizadas, reflejando la creatividad en las formas de manifestación.

# **Tácticas disruptivas y tácticas violentas:**

# Son menos comunes, pero su presencia indica situaciones de mayor tensión.

# **Tácticas autodestructivas:**

# Son las menos reportadas, lo cual es un indicador positivo en términos de la violencia autoinfligida en los conflictos.
#                 """)

    # ------------------------------------------------------------
    # ARRESTOS HERIDOS MUERTOS
    # ------------------------------------------------------------
    st.divider()
    # Relación de Arrestos, Heridos y Muertos a lo Largo de los Años

    # Crear un gráfico de líneas para mostrar la cantidad de Arrestos, Heridos y Muertos a lo largo de los años
    arrestos = df.groupby(
        'año')['Cantidad de arrestos'].sum().reset_index()
    heridos = df.groupby(
        'año')['Manifestantes heridos'].sum().reset_index()
    muertos = df.groupby(
        'año')['Manifestantes muertos'].sum().reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=arrestos['año'], y=arrestos['Cantidad de arrestos'],
                             mode='lines+markers',
                             name='Cantidad de arrestos'))

    fig.add_trace(go.Scatter(x=heridos['año'], y=heridos['Manifestantes heridos'],
                             mode='lines+markers',
                             name='Manifestantes heridos'))

    fig.add_trace(go.Scatter(x=muertos['año'], y=muertos['Manifestantes muertos'],
                             mode='lines+markers',
                             name='Manifestantes muertos'))

    fig.update_layout(
        title={'text': 'Cantidad de Arrestos, Heridos y Muertos a lo Largo de los Años', 'xanchor': 'center',
               'yanchor': 'top', 'y': 0.95, 'x': 0.5, 'font': {'size': 18}},
        xaxis_title='Año',
        yaxis_title='Cantidad',
        width=1000,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        ##### Relación de Arrestos, Heridos y Muertos a lo Largo de los Años
        
El gráfico muestra cómo han variado los arrestos, heridos y muertos en conflictos sociales a lo largo de los años:

**Arrestos:** Hay años en los que el número de arrestos es significativamente mayor.


**Heridos:** La cantidad de heridos muestra variaciones a lo largo de los años, indicando periodos con mayor violencia.


**Muertos:** Afortunadamente, los números son relativamente bajos, pero aún así, cualquier aumento es motivo de preocupación.

Este análisis proporciona una visión clara de las tendencias en la severidad de los conflictos sociales en términos de consecuencias humanas.
                """)

    pro_bar.progress(0.9, text="Construyendo gráficos...")
# --------------------------
with tabTable:
    # --------------------------
    dftabla = df.copy()
    dftabla['fecha'] = dftabla['fecha'].dt.date
    st.dataframe(dftabla, height=600)

    # with st.sidebar:
    #     st.download_button("Descargar datos (csv)",
    #                        convert_df(df), "conflictos.csv",
    #                        "text/csv", key="donwload-csv")
    pro_bar.empty()

# --------------------------
with tabBleau:  # graficos personalizados
    # @st.cache_resource

    def get_pyg_renderer() -> "StreamlitRenderer":
        return StreamlitRenderer(df, spec_io_mode="rw")

    renderer = get_pyg_renderer()
    renderer.explorer()

# --------------------------
# with tabInfo:
#     # --------------------------
#     st.write("Información")

    # eliminar columnas
    # df = df.drop(columns=["p1", "p2a", "p2b", "p2c", "pm",
    #                       "p3", "p5d", "p5e", "p5f",  "p11a",
    #                       "p13b", "p13c", "p13d", "p17b", "p17c", "p17d", "p17e", "p17f", "p18b",
    #                       'p19a1', 'p19a2', 'p19b1', 'p19b2', 'p19c1', 'p19c2', 'p19d1', 'p19d2', 'p19e1', 'p19e2',
    #                       "p24", "p26b",
    #                       "p26d", "p26f", "p28b", "p28d", "p28f", "p29a", "p29b", "p29c", "p29d", "p29e", "p29f"])

    # cambio de nombre de columnas
    # df = df.rename(columns={
    #     'Region': 'Region_id',
    #     'Provincia': 'Provincia_id',
    #     'Comuna': 'Comuna_id',
    #     'pa': 'Tipo medio',
    #     'pb': 'Radial',
    #     'p0': 'Cobertura',
    #     'p3a': 'Tipo noticia',
    #     'p4': 'lineas',
    #     'p5a': 'dia',
    #     'p5b': 'mes',
    #     'p5c': 'año',
    #     'p6': 'Region',
    #     'p7': 'Provincia',
    #     'p8': 'Comuna',
    #     'p9': 'Localidad',
    #     'p9a': 'Urbano',
    #     'p10': 'Lugar objetivo',
    #     'p10a': 'ID evento',
    #     'p11': 'Cantidad participantes',
    #     'p12': 'Estimacion',
    #     'p13a': 'Grupo social',
    #     'p16': 'Organizaciones participantes',
    #     'p17a': 'Actor',
    #     'p20a': 'Conflicto social',
    #     'p20b': 'Conflicto institucional',
    #     'p21': 'Carabineros',
    #     'p23': 'Arrestos',
    #     'p25': 'Heridos',
    #     'p26a': 'Manifestantes heridos',
    #     'p26c': 'Carabineros heridos',
    #     'p26e': 'Personas heridas',
    #     'p28a': 'Manifestantes muertos',
    #     'p28c': 'Carabineros muertos',
    #     'p28e': 'Personas muertas'
    # })

    # Obtener la suma de las ocurrencias de cada tipo de conflicto por mes
