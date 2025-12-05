#src\testRutas.py
"""
testRutas.py
-------------
Pruebas oficiales del Componente 2 del proyecto.

Optimizado:
 - Carga rápida del grafo desde graphml
 - Solo genera CSV si no existen
 - Solo genera gráficas si no existen
 - Evita ejecuciones repetidas de 75 búsquedas
"""

from __future__ import annotations
import os
import csv
import matplotlib.pyplot as plt

import osmnx as ox
from src.problemaRuta import (
    select_node_pairs_by_distance,
    evaluate_algorithms_on_pairs,
)
from src.utils import ensure_dir, save_csv


DATA_DIR = "datos"
ensure_dir(DATA_DIR)

GRAPH_PATH = os.path.join(DATA_DIR, "grafo.graphml")

PLACE = "Tec de Monterrey campus Guadalajara, Zapopan, México"
DIST = 3000     # optimizado
NTYPE = "drive"


# ===============================
# CARGA / GUARDADO DEL GRAFO
# ===============================
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


# ===============================
# GRAFICAR (solo si no existe)
# ===============================
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


# ===============================
# MAIN
# ===============================
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

    print(" SHORT :", short_pairs)
    print(" MEDIUM:", medium_pairs)
    print(" LONG  :", long_pairs)

    # ------------------------------
    # SHORT
    # ------------------------------
    out_short = os.path.join(DATA_DIR, "resultados_short.csv")
    if os.path.exists(out_short):
        print("[testRutas] CSV short existente, cargando…")
        with open(out_short, encoding="utf-8") as f:
            r_short = list(csv.DictReader(f))
    else:
        print("[testRutas] Ejecutando pruebas SHORT…")
        r_short = evaluate_algorithms_on_pairs(G, short_pairs, out_csv=out_short)

    # ------------------------------
    # MEDIUM
    # ------------------------------
    out_medium = os.path.join(DATA_DIR, "resultados_medium.csv")
    if os.path.exists(out_medium):
        print("[testRutas] CSV medium existente, cargando…")
        with open(out_medium, encoding="utf-8") as f:
            r_medium = list(csv.DictReader(f))
    else:
        print("[testRutas] Ejecutando pruebas MEDIUM…")
        r_medium = evaluate_algorithms_on_pairs(G, medium_pairs, out_csv=out_medium)

    # ------------------------------
    # LONG
    # ------------------------------
    out_long = os.path.join(DATA_DIR, "resultados_long.csv")
    if os.path.exists(out_long):
        print("[testRutas] CSV long existente, cargando…")
        with open(out_long, encoding="utf-8") as f:
            r_long = list(csv.DictReader(f))
    else:
        print("[testRutas] Ejecutando pruebas LONG…")
        r_long = evaluate_algorithms_on_pairs(G, long_pairs, out_csv=out_long)

    # ------------------------------
    # CSV COMBINADO
    # ------------------------------
    print("\n[testRutas] Generando CSV combinado…")

    combined = r_short + r_medium + r_long

    out_all = os.path.join(DATA_DIR, "resultados_todo.csv")
    save_csv(combined, out_all)

    print("[testRutas] CSV combinado guardado:", out_all)

    # ------------------------------
    # GRÁFICAS
    # ------------------------------
    print("\n[testRutas] Generando gráficas…")

    graficar_resultados(r_short, "Tiempos SHORT (<1000m)", "grafica_tiempos_short.png")
    graficar_resultados(r_medium, "Tiempos MEDIUM (1000-5000m)", "grafica_tiempos_medium.png")
    graficar_resultados(r_long, "Tiempos LONG (>5000m)", "grafica_tiempos_long.png")
    graficar_resultados(combined, "Tiempos Globales", "grafica_tiempos_global.png")

    # ------------------------------
    # RESUMEN
    # ------------------------------
    print("\n====================================")
    print("       RESUMEN PARA EL REPORTE      ")
    print("====================================")

    def resumen(nombre, results):
        print(f"\n-- {nombre.upper()} --")
        algs = {}
        for r in results:
            alg = r["algorithm"]
            algs.setdefault(alg, []).append(float(r["time_s"]))

        for a, vals in algs.items():
            print(f" {a:5s} promedio = {sum(vals)/len(vals):.4f}s")

    resumen("short", r_short)
    resumen("medium", r_medium)
    resumen("long", r_long)

    print("\n[testRutas] Listo. Script optimizado terminado.\n")
