from __future__ import annotations
import os
import math
import time
import random
import csv

import osmnx as ox
from simpleai.search import (
    breadth_first,
    depth_first,
    uniform_cost,
    astar
)

from src.problemaRuta import crear_problema
from src.utils import ensure_dir, save_csv

# Directorio por defecto para resultados
DATA_DIR = "datos"
ensure_dir(DATA_DIR)

# Función para streamlit
def run_routing_tests(G, num_pairs=5, out_csv="datos/rutas_fast.csv"):
    """
    Ejecuta comparativa rápida de BFS, DFS, UCS y A* sobre un grafo OSMnx
    SIN descargas, SIN gráficas, SOLO guarda un CSV.

    Compatible con Streamlit.
    """

    print("\n[testRutas] Ejecutando prueba rápida STREAMLIT\n")

    nodes = list(G.nodes())
    results = []

    algorithms = [
        ("BFS", breadth_first),
        ("DFS", depth_first),
        ("UCS", uniform_cost),
        ("A*", astar),
    ]

    # Distancia euclidiana aproximada para filtrar pares demasiado lejos
    def distancia(G, n1, n2):
        x1, y1 = G.nodes[n1]["x"], G.nodes[n1]["y"]
        x2, y2 = G.nodes[n2]["x"], G.nodes[n2]["y"]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    count = 0
    intentos = 0
    max_intentos = num_pairs * 30

    while count < num_pairs and intentos < max_intentos:
        intentos += 1
        origen = random.choice(nodes)
        destino = random.choice(nodes)

        # evitar rutas muy grandes para DFS
        try:
            d_aprox = distancia(G, origen, destino)
        except:
            continue

        if origen == destino or d_aprox > 800:
            continue

        count += 1
        print(f"[Par {count}] distancia aprox: {d_aprox:.1f} m")

        problema = crear_problema(G, origen, destino)

        # Corre todos los algoritmos 
        for name, func in algorithms:
            print(f"  -> {name}", end=" ", flush=True)

            t0 = time.time()

            try:
                res = func(problema, graph_search=True)
                elapsed = time.time() - t0

                results.append({
                    "test_id": count,
                    "origen": origen,
                    "destino": destino,
                    "algoritmo": name,
                    "tiempo_s": elapsed,
                    "costo": res.cost if res else None,
                })

                print(f"OK {elapsed:.3f}s")

            except Exception as e:
                print("ERROR")
                results.append({
                    "test_id": count,
                    "origen": origen,
                    "destino": destino,
                    "algoritmo": name,
                    "tiempo_s": -1,
                    "error": str(e),
                })

    # Guardar resultados
    ensure_dir("datos")
    save_csv(results, out_csv)

    print(f"\n[testRutas] Resultados guardados en {out_csv}")
    return out_csv
