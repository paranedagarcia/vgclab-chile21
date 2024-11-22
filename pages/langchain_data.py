from funciones import menu_pages
from dotenv import load_dotenv
import duckdb
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import pandas as pd
import os
import streamlit as st

# Page title
st.set_page_config(page_title='🦜🔗 Consultas de datos')
st.header('🦜🔗 Consultas de datos basados en IA')

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
load_dotenv()
menu_pages()

API_KEY = os.environ['OPENAI_API_KEY']

#
datasets = {'Maceda': 'maceda',
            'Conflictos sociales': 'conflictos',
            'Detenciones': 'detenciones',
            'Licencias médicas': 'licencias'}

with st.expander("Información importante"):
    st.write("Las respuestas son generadas por un modelo de lenguaje de OpenAI, el cual permite realizar consultas sobre el conjunto de datos elegido. Ingrese su consulta la que será respondida por el modelo en forma de texto.")
    st.markdown(
        """Por ejemplo, puede preguntar:
    
¿Cuántos eventos ocurrieron en la región 'araucania' en el año '2018'? y qué tipo de eventos fueron los mas recurrentes? lista los 3 más frecuentes
""")
    st.info(
        "*Nota*: Esta es una tecnología en experimentación por lo que las respuestas pueden no ser del todo exactas.")


@st.cache_data
def cargar_datos(datas):
    # Aquí puedes cargar tus datos desde un archivo CSV, base de datos, etc.
    query = f"SELECT * FROM {datas}"
    dfds = conn.execute(query).fetchdf()
    return dfds


ds1_options = list(datasets.keys())
dataset = st.selectbox("Seleccione el conjunto de datos a consultar",
                       options=ds1_options, key="dataset")


def get_response(df, query):
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-4-turbo-2024-04-09"),
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )
    return agent.invoke(query)


if dataset:
    conn = duckdb.connect('vglab.db')
    query = f"SELECT * FROM {datasets[dataset]}"
    df = conn.execute(query).fetchdf()

    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-4-turbo-2024-04-09"),
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )

    query = st.text_area(
        "Consulta", value="¿Cuántos eventos ocurrieron en la región 'araucania' en el año '2018'? y qué tipo de eventos fueron los mas recurrentes? lista los 3 más frecuentes")
    btn = st.button("Genera respuesta")

    if btn:
        progress_text = "Procesando respuesta..."
        with st.spinner(progress_text):
            # pro_bar = st.progress(0, text=progress_text)
            # pro_bar.progress(0.40, text=progress_text)
            response = get_response(df, query)
            # st.write(response["output"])
            # pro_bar.progress(0.80, text=progress_text)

            st.write(agent.run(query))
        # pro_bar.empty()

        conn.close()
