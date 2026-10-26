"""
DBSCAN Geospatial Hotspot Clustering Engine
Identifies high-density clusters of road potholes using Haversine distance.
Calculates cluster centroids, bounding radius, density, and severity aggregations.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

class PotholeHotspotClusterer:
    def __init__(self, eps_meters=300.0, min_samples=3):
        """
        eps_meters: Maximum distance between two points to be considered in the same neighborhood.
        min_samples: Minimum number of potholes to form a dense municipal hotspot cluster.
        """
        self.eps_meters = eps_meters
        self.min_samples = min_samples
        # Earth radius in meters
        self.EARTH_RADIUS = 6371000.0
        self.eps_radians = self.eps_meters / self.EARTH_RADIUS
        
    def cluster_points(self, df):
        """
        df: DataFrame containing at least ['latitude', 'longitude', 'severity_score']
        """
        if len(df) == 0:
            return df, []
            
        coords = df[['latitude', 'longitude']].values
        coords_rad = np.radians(coords)
        
        # Fit DBSCAN with haversine metric
        db = DBSCAN(
            eps=self.eps_radians,
            min_samples=self.min_samples,
            metric='haversine',
            algorithm='ball_tree'
        )
        labels = db.fit_predict(coords_rad)
        
        clustered_df = df.copy()
        clustered_df['cluster_id'] = labels
        clustered_df['is_hotspot_core'] = labels != -1
        
        # Metrics
        unique_labels = set(labels)
        n_clusters = len(unique_labels - {-1})
        n_noise = int(np.sum(labels == -1))
        
        sil_score = None
        # Silhouette score requires at least 2 clusters and non-noise points
        non_noise_mask = labels != -1
        if n_clusters >= 2 and np.sum(non_noise_mask) > n_clusters:
            try:
                sil_score = round(float(silhouette_score(coords_rad[non_noise_mask], labels[non_noise_mask], metric='haversine')), 4)
            except Exception:
                sil_score = None
                
        # Aggregate cluster characteristics
        cluster_summaries = []
        for cid in sorted(list(unique_labels - {-1})):
            c_points = clustered_df[clustered_df['cluster_id'] == cid]
            centroid_lat = c_points['latitude'].mean()
            centroid_lon = c_points['longitude'].mean()
            
            # Approximate radius in meters from centroid to furthest point
            d_lats = (c_points['latitude'] - centroid_lat) * 111320.0
            d_lons = (c_points['longitude'] - centroid_lon) * 111320.0 * np.cos(np.radians(centroid_lat))
            distances_m = np.sqrt(d_lats**2 + d_lons**2)
            max_radius_m = round(float(distances_m.max()), 1)
            
            pothole_count = len(c_points)
            avg_sev = round(float(c_points['severity_score'].mean()), 1)
            max_sev = round(float(c_points['severity_score'].max()), 1)
            
            # Hotspot priority weighting: density * severity
            intensity = round((pothole_count * avg_sev) / 10.0, 1)
            
            # Dominant ward & road
            ward = c_points['ward_name'].mode()[0] if 'ward_name' in c_points.columns else "Urban Sector"
            road = c_points['road_name'].mode()[0] if 'road_name' in c_points.columns else "Municipal Corridor"
            
            cluster_summaries.append({
                "cluster_id": int(cid),
                "centroid_latitude": round(float(centroid_lat), 6),
                "centroid_longitude": round(float(centroid_lon), 6),
                "pothole_count": pothole_count,
                "radius_meters": max_radius_m,
                "average_severity": avg_sev,
                "max_severity": max_sev,
                "hotspot_intensity": intensity,
                "ward_name": ward,
                "primary_road": road
            })
            
        cluster_df = pd.DataFrame(cluster_summaries)
        
        metrics = {
            "num_clusters": n_clusters,
            "num_noise_points": n_noise,
            "total_points": len(df),
            "noise_ratio": round(n_noise / len(df), 4) if len(df) > 0 else 0.0,
            "silhouette_score": sil_score,
            "clusters": cluster_summaries
        }
        
        return clustered_df, cluster_df, metrics

if __name__ == "__main__":
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gps_path = os.path.join(BASE_DIR, "data", "gps_telemetry", "road_survey_telemetry.csv")
    if os.path.exists(gps_path):
        telemetry = pd.read_csv(gps_path)
        # Filter only records with detected potholes
        pothole_records = telemetry[telemetry['true_potholes'] > 0].copy()
        pothole_records['severity_score'] = pothole_records['max_severity']
        
        clusterer = PotholeHotspotClusterer(eps_meters=350.0, min_samples=3)
        clustered_pts, summary_df, stats = clusterer.cluster_points(pothole_records)
        
        print("\n" + "=" * 50)
        print("DBSCAN HOTSPOT DETECTION RESULTS:")
        print(f"Total Pothole Coordinates Analyzed: {stats['total_points']}")
        print(f"Hotspot Clusters Detected:         {stats['num_clusters']}")
        print(f"Isolated/Noise Incidents:           {stats['num_noise_points']} ({stats['noise_ratio']*100:.1f}%)")
        print(f"Silhouette Score:                   {stats['silhouette_score']}")
        print("=" * 50)
        print("\nCluster Summary Head:")
        print(summary_df.head())
