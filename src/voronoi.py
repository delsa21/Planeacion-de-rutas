# src/voronoi.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
import osmnx as ox
from .utils import haversine_meters

class GestorHospitales:
    def __init__(self, G):
        self.G = G
        self.hospitales = []
        self.coords_hosp = []
        self.tree = None
    
    def cargar_hospitales_reales(self, place, dist=5000):
        """
        Busca hospitales reales en OSM dentro del radio especificado.
        Si falla o no encuentra, usa hospitales ficticios.
        """
        print(f"🏥 Buscando hospitales reales en '{place}' (dist={dist})...")
        tags = {'amenity': ['hospital', 'clinic', 'doctors'], 'building': 'hospital'}
        
        try:
            # Descargar geometrías (hospitales/clínicas)
            gdf = ox.features_from_address(place, tags=tags, dist=dist)
            
            if gdf.empty:
                print("⚠️ No se encontraron hospitales reales. Usando simulados.")
                self.cargar_hospitales_ficticios()
                return

            # Limpiar listas previas
            self.hospitales = []
            self.coords_hosp = []

            # Procesar cada hospital encontrado
            # Usamos los primeros 15 para no saturar el diagrama
            for idx, row in gdf.head(15).iterrows():
                # Obtener geometría y su centroide
                geom = row.geometry
                centroid = geom.centroid
                lat, lon = centroid.y, centroid.x
                
                # Obtener nombre (si existe)
                nombre = row.get('name', 'Centro de Salud sin nombre')
                if not isinstance(nombre, str): nombre = "Hospital (Genérico)"

                # Encontrar el nodo del grafo más cercano al hospital (Requisito PDF)
                # Esto conecta el mundo real (lat/lon) con tu grafo
                node_cercano = ox.distance.nearest_nodes(self.G, lon, lat)
                
                # Guardar datos
                self.hospitales.append({
                    "id": node_cercano,
                    "lat": lat, 
                    "lon": lon, 
                    "nombre": nombre
                })
                self.coords_hosp.append((lat, lon))

            print(f"✅ Se encontraron {len(self.hospitales)} hospitales reales.")
            
            # Construir KDTree para búsquedas rápidas
            if self.coords_hosp:
                self.tree = KDTree(np.array(self.coords_hosp))
            else:
                self.cargar_hospitales_ficticios()

        except Exception as e:
            print(f"⚠️ Error descargando hospitales: {e}. Usando simulados.")
            self.cargar_hospitales_ficticios()

    def cargar_hospitales_ficticios(self, n=5):
        """Elige n nodos al azar para ser hospitales (Método de respaldo)"""
        print(f"🎲 Generando {n} hospitales simulados...")
        nodes = list(self.G.nodes(data=True))
        # Aseguramos no pedir más nodos de los que existen
        n = min(n, len(nodes))
        indices = np.random.choice(len(nodes), n, replace=False)
        
        self.hospitales = []
        self.coords_hosp = []
        
        for i, idx in enumerate(indices):
            nid, data = nodes[idx]
            # Manejo defensivo de coordenadas
            lat = data.get('y', data.get('lat'))
            lon = data.get('x', data.get('lon'))
            
            self.hospitales.append({
                "id": nid, "lat": lat, "lon": lon, "nombre": f"Hospital Simulado #{i+1}"
            })
            self.coords_hosp.append((lat, lon))
            
        self.tree = KDTree(np.array(self.coords_hosp))

    def encontrar_hospital_cercano(self, lat, lon):
        """Usa KDTree para encontrar el hospital más cercano (equiv. Voronoi)"""
        if not self.tree: return None, 0
        dist_deg, idx = self.tree.query((lat, lon))
        hosp = self.hospitales[idx]
        dist_m = haversine_meters((lat, lon), (hosp['lat'], hosp['lon']))
        return hosp, dist_m

    def visualizar_voronoi(self):
        """Pinta el diagrama de Voronoi"""
        if len(self.coords_hosp) < 3: 
            print("⚠️ Se necesitan al menos 3 hospitales para Voronoi.")
            return None
        
        points = np.array(self.coords_hosp)
        vor = Voronoi(points)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        # Plotear diagrama
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2)
        
        # Dibujar los puntos de los hospitales
        ax.plot(points[:, 1], points[:, 0], 'b^', markersize=8, label='Hospitales')
        
        # Opcional: poner etiquetas
        # for h in self.hospitales:
        #     ax.text(h['lon'], h['lat'], h['nombre'], fontsize=8)

        plt.title("Regiones de Voronoi (Hospitales)")
        plt.xlabel("Longitud")
        plt.ylabel("Latitud")
        plt.legend()
        return fig