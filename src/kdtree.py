# src/kdtree.py
from __future__ import annotations
import os
import time
from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial import KDTree
import osmnx as ox
import networkx as nx
from shapely.geometry import Point
import pandas as pd
from .utils import (
    haversine_meters,
    timeit,
    ensure_dir,
    save_csv,
    format_seconds,
)

# directorio por defecto para resultados
DEFAULT_DATA_DIR = "datos"
ensure_dir(DEFAULT_DATA_DIR)


# cargar grafo osm alrededor de place
def load_graph(
    place: str, dist: int = 10000, network_type: str = "drive", simplify: bool = True
) -> nx.MultiDiGraph:
    if not isinstance(place, str) or not place.strip():
        raise ValueError("place debe ser una cadena no vacía.")
    print(
        f"[kdtree] Descargando grafo para '{place}' (dist={dist}, tipo={network_type})..."
    )
    G = ox.graph_from_address(
        place, dist=dist, network_type=network_type, simplify=simplify
    )
    print(f"[kdtree] Grafo cargado: nodos={len(G.nodes)} aristas={len(G.edges)}")
    return G


# reproyecta a crs metrico si se solicita
def project_graph_if_needed(
    G: nx.MultiDiGraph, force_project: bool = True
) -> nx.MultiDiGraph:
    if not force_project:
        return G
    return ox.project_graph(G)


# extrae nodos y coordenadas, proyectadas o latlon
def extract_nodes_coords(
    G: nx.Graph, use_projected: bool = False
) -> Tuple[List[int], List[Tuple[float, float]]]:
    nodes = []
    coords = []
    for nid, data in G.nodes(data=True):
        nodes.append(nid)
        if use_projected:
            x = data.get("x", None)
            y = data.get("y", None)
            if x is None or y is None:
                y = data.get("y", data.get("lat", None))
                x = data.get("x", data.get("lon", None))
        else:
            y = data.get("y", data.get("lat", None))
            x = data.get("x", data.get("lon", None))
        if x is None or y is None:
            raise RuntimeError(f"Nodo {nid} no tiene coordenadas válidas.")
        coords.append((float(y), float(x)))
    return nodes, coords


# construye el kdtree
@timeit
def build_kdtree(coords: List[Tuple[float, float]]):
    if not coords:
        raise ValueError("coords vacía.")
    arr = np.array(coords)
    tree = KDTree(arr)
    return tree


# deteccion simple para saber si coords parecen latlon
def _coords_look_like_latlon(coords: List[Tuple[float, float]]) -> bool:
    try:
        lat_vals = [abs(p[0]) for p in coords]
        return max(lat_vals) <= 90
    except Exception:
        return False


# busqueda con kdtree
def nearest_kdtree(
    tree: KDTree,
    coords: List[Tuple[float, float]],
    nodes: List[int],
    query_point: Tuple[float, float],
    k: int = 1,
):
    if k < 1:
        raise ValueError("k debe ser >=1")
    dist, idx = tree.query(query_point, k=k)
    if k == 1:
        idxs = [int(idx)]
        dists = [float(dist)]
    else:
        idxs = [int(i) for i in idx]
        dists = [float(d) for d in dist]

    if _coords_look_like_latlon(coords):
        dists_m = [haversine_meters(query_point, coords[i]) for i in idxs]
    else:
        dists_m = dists

    node_ids = [nodes[i] for i in idxs]
    return node_ids, dists_m


# busqueda por fuerza bruta usando haversine
def nearest_bruteforce(
    coords: List[Tuple[float, float]],
    nodes: List[int],
    query_point: Tuple[float, float],
):
    best_idx = None
    best_d = float("inf")
    for i, pt in enumerate(coords):
        d = haversine_meters(query_point, pt)
        if d < best_d:
            best_d = d
            best_idx = i
    return nodes[best_idx], best_d


# wrappers temporizados
@timeit
def nearest_kdtree_timed(tree, coords, nodes, query_point, k=1):
    return nearest_kdtree(tree, coords, nodes, query_point, k)


@timeit
def nearest_bruteforce_timed(coords, nodes, query_point):
    return nearest_bruteforce(coords, nodes, query_point)


# genera puntos de prueba
def sample_query_points(
    coords: List[Tuple[float, float]], n: int = 20, mode: str = "first"
):
    if n <= 0:
        return []
    if mode == "first":
        return coords[:n]
    if mode == "random":
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(coords), size=min(n, len(coords)), replace=False)
        return [coords[i] for i in idxs]
    if mode == "jitter":
        rng = np.random.default_rng(0)
        res = []
        for i in range(min(n, len(coords))):
            lat, lon = coords[i]
            lat += rng.normal(0, 1e-4)
            lon += rng.normal(0, 1e-4)
            res.append((lat, lon))
        return res
    raise ValueError("mode no reconocido")


# pruebas comparativas kdtree vs brute force
def run_kdtree_tests(
    G: nx.Graph,
    n_tests: int = 20,
    sample_mode: str = "first",
    out_csv: Optional[str] = None,
    use_projected: bool = True,
):
    if out_csv is None:
        out_csv = os.path.join(DEFAULT_DATA_DIR, "kdtree_tests.csv")

    if use_projected:
        G_proj = project_graph_if_needed(G, force_project=True)
        nodes, coords = extract_nodes_coords(G_proj, use_projected=True)
        projected = True
    else:
        nodes, coords = extract_nodes_coords(G, use_projected=False)
        projected = False

    (tree, t_build) = build_kdtree(coords)
    print(f"[kdtree] Tiempo construcción: {format_seconds(t_build)}")

    queries = sample_query_points(coords, n=n_tests, mode=sample_mode)
    rows = []
    for i, q in enumerate(queries, start=1):
        (res_kdt, t_kdt) = nearest_kdtree_timed(tree, coords, nodes, q, k=1)
        node_kdt = res_kdt[0][0] if isinstance(res_kdt[0], list) else res_kdt[0]
        d_kdt = res_kdt[1][0] if isinstance(res_kdt[1], list) else res_kdt[1]

        (res_bf, t_bf) = nearest_bruteforce_timed(coords, nodes, q)
        node_bf, d_bf = res_bf

        consistent = node_kdt == node_bf

        row = {
            "test_id": i,
            "query_lat": float(q[0]),
            "query_lon": float(q[1]),
            "kdtree_node": int(node_kdt),
            "kdtree_dist_m": float(d_kdt),
            "kdtree_time_s": float(t_kdt),
            "bruteforce_node": int(node_bf),
            "bruteforce_dist_m": float(d_bf),
            "bruteforce_time_s": float(t_bf),
            "consistent_node": bool(consistent),
            "coords_projected": bool(projected),
            "kdtree_build_time_s": float(t_build),
        }
        rows.append(row)

        print(
            f"[kdtree][Test {i}] Kdtree {format_seconds(t_kdt)} | Brute {format_seconds(t_bf)} | Consistent={consistent}"
        )

    save_csv(rows, out_csv)
    print(f"[kdtree] Resultados guardados en: {out_csv}")
    return rows


# wrapper para compatibilidad con nombre anterior
def run_tests_kdtree(
    G: nx.Graph,
    n_tests: int = 20,
    sample_mode: str = "first",
    out_csv: Optional[str] = None,
    use_projected: bool = True,
):
    return run_kdtree_tests(
        G,
        n_tests=n_tests,
        sample_mode=sample_mode,
        out_csv=out_csv,
        use_projected=use_projected,
    )


# demo local
if __name__ == "__main__":
    PLACE = "Tec de Monterrey campus Guadalajara, Zapopan, Jalisco, México"
    G = load_graph(PLACE, dist=5000, network_type="drive")
    rows = run_kdtree_tests(
        G,
        n_tests=10,
        sample_mode="first",
        out_csv=os.path.join(DEFAULT_DATA_DIR, "kdtree_demo.csv"),
        use_projected=True,
    )
    times_kdt = [r["kdtree_time_s"] for r in rows]
    times_bf = [r["bruteforce_time_s"] for r in rows]
    import numpy as _np

    print(
        f"\nResumen (n={len(rows)}): Kdtree avg={_np.mean(times_kdt):.6f}s | Brute avg={_np.mean(times_bf):.6f}s"
    )
