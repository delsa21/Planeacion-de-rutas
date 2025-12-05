"""
testRutas.py
-------------
Incluye dos modos:

1. MODO OFFLINE (if __name__ == "__main__")
   - Descarga / carga grafo graphml
   - Selecciona rutas por buckets short/medium/long
   - Ejecuta 75 pruebas
   - Genera CSV y gráficas

2. MODO STREAMLIT (función run_routing_tests)
   - Usa grafo ya cargado
   - Genera pruebas rápidas BFS/DFS/UCS/A*
   - Crea CSV simple sin gráficas
"""

from __future__ import annotations
import os
import csv
import time
import math
import random
import matplotlib.pyplot as plt

import osmnx as ox
from simpleai.search import (
    breadth_first,
    depth_first,
    uniform_cost,
    astar
)

# ======================================================
# IMPORTS DEL PROYECTO
# ======================================================
from src.problemaRuta import (
    select_node_pairs_by_distance,
    evaluate_algorithms_on_pairs,
    crear_problema
)
from src.utils import ensure_dir, save_csv


# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================
DATA_DIR = "datos"
ensure_dir(DATA_DIR)

GRAPH_PATH = os.path.join(DATA_DIR, "grafo.graphml")

PLACE = "Tec de Monterrey campus Guadalajara, Zapopan, México"
DIST = 3000
NTYPE = "drive"


# ======================================================
# FUNCIÓN STREAMLIT
# ======================================================
def run_routing_tests(G, num_pairs=5, out_csv="datos/rutas_fast.csv"):
    """
    Función simplificada y compatible con Streamlit.
    Ejecuta comparativa rápida de BFS, DFS, UCS y A*.

    - No descarga mapas
    - No genera gráficas
    - Solo guarda un CSV
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

    # seguridad para evitar rutas enormes con DFS
    def distancia(G, n1, n2):
        x1, y1 = G.nodes[n1]["x"], G.nodes[n1]["y"]
        x2, y2 = G.nodes[n2]["x"], G.nodes[n2]["y"]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    count = 0
    intentos = 0
    max_intentos = num_pairs * 30

    while count < num_pairs and intentos < max_intentos:
        intentos += 1
        o = random.choice(nodes)
        d = random.choice(nodes)

        # filtro rápido
        try:
            d_aprox = distancia(G, o, d)
        except:
            continue

        if o == d or d_aprox > 800:
            continue

        count += 1
        print(f"[Par {count}] distancia aprox: {d_aprox:.1f} m")

        problema = crear_problema(G, o, d)

        for name, func in algorithms:
            print(f"  -> {name}", end=" ", flush=True)
            t0 = time.time()

            try:
                res = func(problema, graph_search=True)
                t = time.time() - t0

                results.append({
                    "test_id": count,
                    "origen": o,
                    "destino": d,
                    "algoritmo": name,
                    "tiempo_s": t,
                    "costo": res.cost if res else None,
                })

                print(f"OK {t:.2f}s")

            except Exception as e:
                print("ERROR")
                results.append({
                    "test_id": count,
                    "origen": o,
                    "destino": d,
                    "algoritmo": name,
                    "tiempo_s": -1,
                    "error": str(e),
                })

    ensure_dir("datos")
    save_csv(results, out_csv)

    print(f"\n[testRutas] Resultados guardados en {out_csv}")
    return out_csv



# ======================================================
# MODO OFFLINE FULL (original)
# ======================================================
def cargar_grafo():
    if os.path.exists(GRAPH_PATH):
        print("[testRutas] Cargando grafo desde archivo...")
        G = ox.load_graphml(GRAPH_PATH)
        print("[testRutas] Grafo cargado:", len(G.nodes), "nodos")
        return G

    print("[testRutas] Descargando grafo por primera vez…")
    G = ox.graph_from_address(PLACE, dist=DIST, network_type=NTYPE)
    ox.save_graphml(G, GRAPH_PATH)
    print("[testRutas] Grafo descargado y guardado:", len(G.nodes), "nodos")
    return G


def graficar_resultados(df, titulo, nombre_archivo):
    out_path = os.path.join(DATA_DIR, nombre_archivo)

    if os.path.exists(out_path):
        print(f"[testRutas] Gráfica existente, saltando: {nombre_archivo}")
        return

    plt.figure(figsize=(10, 6))

    algoritmos = [r["algorithm"] for r in df]
    tiempos = [float(r["time_s"]) for r in df]

    plt.bar(algoritmos, tiempos)
    plt.title(titulo)
    plt.ylabel("Tiempo (s)")
    plt.xlabel("Algoritmo")
    plt.grid(axis="y", alpha=0.3)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[testRutas] Gráfica generada: {nombre_archivo}")


# ======================================================
# EJECUCIÓN COMO SCRIPT (offline normal)
# ======================================================
if __name__ == "__main__":
    print("\n====================================")
    print("     TEST DE ALGORITMOS OPTIMIZADO  ")
    print("====================================\n")

    G = cargar_grafo()

    print("[testRutas] Seleccionando pares…")
    buckets = select_node_pairs_by_distance(G, n_pairs_per_bucket=5)

    short_pairs = buckets["short"]
    medium_pairs = buckets["medium"]
    long_pairs = buckets["long"]

    # ------------------------------
    # SHORT
    # ------------------------------
    out_short = os.path.join(DATA_DIR, "resultados_short.csv")
    if os.path.exists(out_short):
        with open(out_short, encoding="utf-8") as f:
            r_short = list(csv.DictReader(f))
    else:
        r_short = evaluate_algorithms_on_pairs(G, short_pairs, out_csv=out_short)

    # ------------------------------
    # MEDIUM
    # ------------------------------
    out_medium = os.path.join(DATA_DIR, "resultados_medium.csv")
    if os.path.exists(out_medium):
        with open(out_medium, encoding="utf-8") as f:
            r_medium = list(csv.DictReader(f))
    else:
        r_medium = evaluate_algorithms_on_pairs(G, medium_pairs, out_csv=out_medium)

    # ------------------------------
    # LONG
    # ------------------------------
    out_long = os.path.join(DATA_DIR, "resultados_long.csv")
    if os.path.exists(out_long):
        with open(out_long, encoding="utf-8") as f:
            r_long = list(csv.DictReader(f))
    else:
        r_long = evaluate_algorithms_on_pairs(G, long_pairs, out_csv=out_long)

    # ------------------------------
    # COMBINADO & GRÁFICAS
    # ------------------------------
    combined = r_short + r_medium + r_long

    out_all = os.path.join(DATA_DIR, "resultados_todo.csv")
    save_csv(combined, out_all)

    graficar_resultados(r_short, "SHORT (<1000m)", "grafica_short.png")
    graficar_resultados(r_medium, "MEDIUM (1000-5000m)", "grafica_medium.png")
    graficar_resultados(r_long, "LONG (>5000m)", "grafica_long.png")
    graficar_resultados(combined, "GLOBAL", "grafica_global.png")

    print("\n[testRutas] Script offline terminado.\n")
