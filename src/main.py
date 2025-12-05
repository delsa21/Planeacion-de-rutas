import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import sys
import osmnx as ox
from simpleai.search import astar

# Ajustar path para importar módulos de src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importamos tus módulos
from src.kdtree import load_graph, extract_nodes_coords, run_tests_kdtree
from src.testRutas import run_routing_tests
from src.voronoi import GestorHospitales
from src.problemaRuta import crear_problema

# Interfaz
st.set_page_config(page_title="Rutas Rápidas", layout="centered")
st.title("Proyecto: Planeación de Rutas")
st.header("1. Cargar el Mapa")

# Variable global para el lugar, para usarla en Voronoi también
PLACE_DEFAULT = "Tec de Monterrey campus Guadalajara, Zapopan, Jalisco, México"

if 'mapa_listo' not in st.session_state:
    st.session_state['mapa_listo'] = False

if st.button("Cargar Mapa de Zapopan"):
    with st.spinner("Descargando mapa (esto puede tardar unos segundos)..."):
        G = load_graph(PLACE_DEFAULT, dist=5000) 
        nodes, coords = extract_nodes_coords(G)
        
        st.session_state['grafo'] = G
        st.session_state['nodos'] = nodes
        st.session_state['coords'] = coords
        st.session_state['mapa_listo'] = True
    
    st.success(f"¡Mapa cargado correctamente! ({len(nodes)} nodos)")

# Menu de opciones
if st.session_state['mapa_listo']:
    
    G = st.session_state['grafo']
    nodes = st.session_state['nodos']
    coords = st.session_state['coords']

    st.divider()
    opcion = st.selectbox(
        "Selecciona una actividad:", 
        ["Selecciona...", "Pruebas KD-Tree", "Comparar Algoritmos", "Servicio de Emergencias (Voronoi)", "Ver Ruta Visual"]
    )

    # Pruebas KD-Tree 
    if opcion == "Pruebas KD-Tree":
        st.subheader("Rendimiento de Búsqueda (KD-Tree vs Fuerza Bruta)")
        st.write("Compara el tiempo de búsqueda del nodo más cercano a una coordenada.")
        
        if st.button("Ejecutar Benchmark"):
            # Llama a la función de test en kdtree.py
            run_tests_kdtree(G, n_tests=20, out_csv="datos/kdtree_fast.csv")
            
            try:
                df = pd.read_csv("datos/kdtree_fast.csv")
                st.dataframe(df)
                
                # Graficar tiempos si las columnas existen
                cols_tiempo = [c for c in df.columns if 'time' in c]
                if len(cols_tiempo) >= 2:
                    st.line_chart(df[cols_tiempo])
                
                st.info("Nota: Los tiempos pueden ser 0.0 si la muestra es pequeña y la CPU muy rápida.")
            except Exception as e:
                st.error(f"Error leyendo resultados: {e}")

    # Comprar Algoritmos
    elif opcion == "Comparar Algoritmos":
        st.subheader("Comparativa (BFS, DFS, A*, IDDFS, UCS)")
        st.write("Evalúa diferentes estrategias de búsqueda entre pares de nodos.")
        st.info("Nota: Se probarán rutas cortas para agilidad en la demo.")

        if st.button("Calcular Rutas"):
            with st.spinner("Procesando algoritmos..."):
                run_routing_tests(G, num_pairs=3, out_csv="datos/rutas_fast.csv")
                
                try:
                    df = pd.read_csv("datos/rutas_fast.csv")
                    st.dataframe(df)
                except:
                    st.error("No se generaron datos. Verifica testRutas.py.")

    # Voronoi
    elif opcion == "Servicio de Emergencias (Voronoi)":
        st.subheader("Hospitales Cercanos")
        st.markdown("""
        Este módulo busca hospitales reales en OSM y divide el mapa en regiones de Voronoi.
        Cualquier emergencia dentro de una región será asignada al hospital correspondiente.
        """)
        
        usar_reales = st.checkbox("Buscar hospitales reales en OpenStreetMap", value=True)
        
        if st.button("Generar Diagrama de Voronoi"):
            gh = GestorHospitales(G)
            
            if usar_reales:
                with st.spinner("Descargando hospitales reales..."):
                    gh.cargar_hospitales_reales(PLACE_DEFAULT, dist=4500)
            else:
                gh.cargar_hospitales_ficticios(n=5)
            
            # Simular emergencia en la posición del primer nodo (solo como ejemplo)
            if len(gh.hospitales) > 0:
                # Tomamos un punto arbitrario del mapa como "ubicación del usuario"
                idx_random = random.randint(0, len(coords)-1)
                t_lat, t_lon = coords[idx_random]
                
                hosp, dist = gh.encontrar_hospital_cercano(t_lat, t_lon)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.error(f"Emergencia reportada en:\nLat: {t_lat:.4f}, Lon: {t_lon:.4f}")
                with col2:
                    st.success(f"Ir al hospital:\n**{hosp['nombre']}**\nDistancia: {dist:.0f} metros")
                
                # Gráfica
                fig = gh.visualizar_voronoi()
                if fig:
                    st.pyplot(fig)
            else:
                st.warning("No se encontraron hospitales para generar el diagrama.")

    # Ruta Visual
    elif opcion == "Ver Ruta Visual":
        st.subheader("Visualizador de Rutas (A*)")
        if st.button("Generar Ruta Aleatoria"):
            origen = random.choice(nodes)
            destino = random.choice(nodes)
            
            st.write(f"Calculando ruta de nodo {origen} a {destino}...")
            problema = crear_problema(G, origen, destino)
            resultado = astar(problema, graph_search=True)
            
            if resultado:
                st.success(f"¡Ruta encontrada! Costo: {resultado.cost:.1f}m")
                camino = [s for a, s in resultado.path()]
                
                # Graficar sobre el mapa
                fig, ax = ox.plot_graph_route(G, camino, node_size=0, bgcolor='white', show=False, close=False)
                st.pyplot(fig)
            else:
                st.warning("Puntos desconectados o inalcanzables. Intenta de nuevo.")

else:
    st.info("Por favor, carga el mapa usando el botón superior para empezar.")