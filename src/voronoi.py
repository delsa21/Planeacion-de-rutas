# src/voronoi.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
from .utils import haversine_meters

class GestorHospitales:
    def __init__(self, G):
        self.G = G
        self.hospitales = []
        self.coords_hosp = []
        self.tree = None
    
    def cargar_hospitales_ficticios(self, n=5):
        """Elige n nodos al azar para ser hospitales"""
        nodes = list(self.G.nodes(data=True))
        indices = np.random.choice(len(nodes), n, replace=False)
        
        for i, idx in enumerate(indices):
            nid, data = nodes[idx]
            lat, lon = data['y'], data['x']
            self.hospitales.append({
                "id": nid, "lat": lat, "lon": lon, "nombre": f"Hospital #{i+1}"
            })
            self.coords_hosp.append((lat, lon))
            
        self.tree = KDTree(np.array(self.coords_hosp))
        print(f"{n} hospitales asignados.")

    def encontrar_hospital_cercano(self, lat, lon):
        """Usa KDTree para encontrar el hospital más cercano (equiv. Voronoi)"""
        if not self.tree: return None, 0
        dist_deg, idx = self.tree.query((lat, lon))
        hosp = self.hospitales[idx]
        dist_m = haversine_meters((lat, lon), (hosp['lat'], hosp['lon']))
        return hosp, dist_m

    def visualizar_voronoi(self):
        """Pinta el diagrama de Voronoi"""
        if len(self.coords_hosp) < 3: return
        points = np.array(self.coords_hosp)
        vor = Voronoi(points)
        voronoi_plot_2d(vor, show_vertices=False, line_colors='orange')
        plt.title("Regiones de Voronoi (Hospitales)")
        plt.show()