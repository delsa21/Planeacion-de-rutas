#src\utils.py
"""
Utilidades compartidas para el proyecto de planeación de rutas.
Funciones:
 - haversine_meters(a,b): distancia geodésica en metros entre dos (lat,lon)
 - timeit(func): decorador que devuelve (resultado, tiempo_segundos)
 - ensure_dir(path): crea directorio si no existe
 - save_csv(table, path): guarda lista de dicts o dict->list en CSV
 - load_csv(path): carga CSV en pandas.DataFrame
 - format_seconds(s): cadena legible de tiempo
 - validate_node_in_graph(G, node): raise error si no existe
"""

from functools import wraps
import time
import os
import csv
from math import radians, sin, cos, sqrt, atan2

import pandas as pd

def haversine_meters(a, b):
    """
    Calcula distancia en metros entre a=(lat,lon) y b=(lat,lon)
    Implementación Haversine (no depende de geopy, es determinista y rápido).
    """
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371000.0  # radio de la tierra en metros
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    aa = sin(dphi/2.0)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2.0)**2
    c = 2 * atan2(sqrt(aa), sqrt(max(0.0, 1.0-aa)))
    return R * c

def timeit(func):
    """
    Decorador: devuelve (resultado, tiempo_en_segundos)
    Si la función ya devolviera una tupla (res, t) no lo modifica.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        res = func(*args, **kwargs)
        t1 = time.time()
        return res, (t1 - t0)
    return wrapper

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_csv(table, path, fieldnames=None):
    """
    Guarda 'table' en CSV.
    - table: lista de dicts OR pandas.DataFrame
    - path: ruta de salida
    """
    ensure_dir(os.path.dirname(path) or ".")
    if isinstance(table, pd.DataFrame):
        table.to_csv(path, index=False)
        return
    if not table:
        # crea archivo vacío con campo 'empty'
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['empty'])
        return
    # infer fieldnames si no se dieron
    if fieldnames is None:
        # union de claves
        keys = set()
        for row in table:
            keys.update(row.keys())
        fieldnames = list(keys)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in table:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

def load_csv(path):
    return pd.read_csv(path)

def format_seconds(s):
    if s is None:
        return "N/A"
    if s < 1.0:
        return f"{s*1000:.1f} ms"
    if s < 60.0:
        return f"{s:.3f} s"
    minutes = int(s // 60)
    seconds = s - 60*minutes
    return f"{minutes} min {seconds:.1f} s"

def validate_node_in_graph(G, node):
    if node not in G.nodes:
        raise ValueError(f"El nodo {node} no existe en el grafo.")

# Self-test simple (se puede ejecutar: python src/utils.py)
if __name__ == "__main__":
    print("Auto-test de utils.py")
    a = (20.6769, -103.344)
    b = (20.6900, -103.350)
    d = haversine_meters(a,b)
    print("dist (m):", d)
    @timeit
    def busy(n):
        s = 0
        for i in range(n):
            s += i*i
        return s
    (res, t) = busy(100000)
    print("busy time:", format_seconds(t), "resultado parcial:", res % 1000)
