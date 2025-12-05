# src/voronoi.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
import osmnx as ox
from .utils import haversine_meters


class GestorHospitales:
    def __init__(self, G):
        # almacena el grafo y estructuras para hospitales
        self.G = G
        self.hospitales = []
        self.coords_hosp = []
        self.tree = None

    def cargar_hospitales_reales(self, place, dist=5000):
        # busca hospitales reales en osm y toma sus centroides
        tags = {"amenity": ["hospital", "clinic", "doctors"], "building": "hospital"}

        try:
            gdf = ox.features_from_address(place, tags=tags, dist=dist)

            if gdf.empty:
                self.cargar_hospitales_ficticios()
                return

            self.hospitales = []
            self.coords_hosp = []

            # se usan solo los primeros 15 hospitales para evitar saturación gráfica
            for _, row in gdf.head(15).iterrows():
                geom = row.geometry
                centroid = geom.centroid
                lat, lon = centroid.y, centroid.x

                nombre = row.get("name", "centro de salud")
                if not isinstance(nombre, str):
                    nombre = "centro de salud"

                # nodo cercano en el grafo
                node_cercano = ox.distance.nearest_nodes(self.G, lon, lat)

                self.hospitales.append(
                    {"id": node_cercano, "lat": lat, "lon": lon, "nombre": nombre}
                )
                self.coords_hosp.append((lat, lon))

            if self.coords_hosp:
                self.tree = KDTree(np.array(self.coords_hosp))
            else:
                self.cargar_hospitals_ficticios()

        except Exception:
            self.cargar_hospitales_ficticios()

    def cargar_hospitales_ficticios(self, n=5):
        # método de respaldo que usa nodos aleatorios del grafo
        nodes = list(self.G.nodes(data=True))
        n = min(n, len(nodes))
        indices = np.random.choice(len(nodes), n, replace=False)

        self.hospitales = []
        self.coords_hosp = []

        for i, idx in enumerate(indices):
            nid, data = nodes[idx]
            lat = data.get("y", data.get("lat"))
            lon = data.get("x", data.get("lon"))

            self.hospitales.append(
                {
                    "id": nid,
                    "lat": lat,
                    "lon": lon,
                    "nombre": f"hospital simulado {i+1}",
                }
            )
            self.coords_hosp.append((lat, lon))

        self.tree = KDTree(np.array(self.coords_hosp))

    def encontrar_hospital_cercano(self, lat, lon):
        # usa kdtree para encontrar hospital más cercano
        if not self.tree:
            return None, 0
        dist_deg, idx = self.tree.query((lat, lon))
        hosp = self.hospitales[idx]
        dist_m = haversine_meters((lat, lon), (hosp["lat"], hosp["lon"]))
        return hosp, dist_m

    def visualizar_voronoi(self):
        # construye y muestra el diagrama de voronoi
        if len(self.coords_hosp) < 3:
            return None

        points = np.array(self.coords_hosp)
        vor = Voronoi(points)

        fig, ax = plt.subplots(figsize=(8, 6))

        voronoi_plot_2d(
            vor,
            ax=ax,
            show_vertices=False,
            line_colors="orange",
            line_width=2,
        )

        ax.plot(points[:, 1], points[:, 0], "b^", markersize=8, label="hospitales")

        plt.title("regiones de voronoi")
        plt.xlabel("longitud")
        plt.ylabel("latitud")
        plt.legend()
        return fig
