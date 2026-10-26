"""
Urban Problem Hotspot Prediction and Decision Support System
Pothole Detection, Physical Quantification, and Spatial Hotspot Analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# Dynamically set BASE_DIR to the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from urban_system.detector.pothole_detector import PotholeDetector
from urban_system.clustering.hotspot_clustering import PotholeHotspotClusterer

st.set_page_config(
    page_title="Urban Problem Hotspot Prediction System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling for clean, professional engineering look
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    .main-title {
        font-size: 26px;
        font-weight: 700;
        color: var(--text-color, #ffffff);
        margin-bottom: 4px;
        letter-spacing: -0.3px;
    }
    .sub-title {
        font-size: 14px;
        color: var(--text-color, #ffffff);
        opacity: 0.82;
        margin-bottom: 20px;
    }
    .metric-container {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 6px;
        padding: 10px 14px;
    }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: var(--text-color, #ffffff);
        margin-top: 10px;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.25);
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Cache system components
@st.cache_resource
def load_detector():
    model_weights = os.path.join(BASE_DIR, "models", "pothole_yolov8_best.pt")
    if not os.path.exists(model_weights):
        model_weights = os.path.join(BASE_DIR, "models", "yolov8n.pt")
    return PotholeDetector(model_weights)

@st.cache_data
def load_telemetry_data():
    csv_path = os.path.join(BASE_DIR, "data", "gps_telemetry", "road_survey_telemetry.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

# Initialize objects
detector = load_detector()
telemetry_df = load_telemetry_data()

# Main Header
st.markdown('<div class="main-title">AI-Based Urban Problem Hotspot Prediction & Decision Support System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Road Infrastructure Monitoring: Deep Learning Distress Identification & Spatial Cluster Modeling</div>', unsafe_allow_html=True)

# Summary Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)
total_surveyed = len(telemetry_df)
potholes_found = int(telemetry_df['true_potholes'].sum()) if not telemetry_df.empty else 0
hard_negatives = int(telemetry_df['is_hard_negative'].sum()) if not telemetry_df.empty else 0
avg_speed = round(telemetry_df['vehicle_speed_kmh'].mean(), 1) if not telemetry_df.empty else 0.0

col1.metric("Survey Frames", f"{total_surveyed}")
col2.metric("Detected Potholes", f"{potholes_found}")
col3.metric("Negative Samples Screened", f"{hard_negatives}", help="Asphalt repair patches, shadows, and surface cracks verified")
col4.metric("Mean Survey Speed", f"{avg_speed} km/h")
col5.metric("Surveyed Corridors", "5 Wards")

st.markdown("<hr style='margin: 12px 0 18px 0; border: none; border-top: 1px solid rgba(128, 128, 128, 0.25);'>", unsafe_allow_html=True)

# Navigation Tabs: ONLY YOLOv8 and DBSCAN without emojis
tab1, tab2 = st.tabs([
    "YOLOv8 Pothole Detection & Quantification",
    "DBSCAN Geospatial Hotspot Map"
])

# =========================================================================
# TAB 1: YOLOV8 POTHOLE DETECTION & QUANTIFICATION (VERTICAL LAYOUT)
# =========================================================================
with tab1:
    # 1. Image Selection / Upload Section
    st.markdown('<div class="section-header">Select or Upload Road Imagery</div>', unsafe_allow_html=True)
    
    col_input1, col_input2 = st.columns([1, 2])
    
    with col_input1:
        img_source = st.radio("Image Source:", ["Standard Survey Samples", "Upload Image File"], horizontal=True)
        
        selected_img_path = None
        input_pil_img = None
        
        if img_source == "Standard Survey Samples":
            sample_dir = os.path.join(BASE_DIR, "data", "pothole_dataset", "images", "val")
            available_samples = sorted(os.listdir(sample_dir)) if os.path.exists(sample_dir) else []
            
            preset_labels = {
                "rdd_val_0000.jpg": "Sample 01: Severe Surface Crater (Daylight)",
                "rdd_val_0002.jpg": "Sample 02: Waterlogged Pothole (Wet Pavement)",
                "rdd_val_0005.jpg": "Sample 03: Pavement Distress with Canopy Shadows",
                "rdd_val_0003.jpg": "Sample 04: Bitumen Repair Patch & Cracks (Control)",
                "rdd_val_0007.jpg": "Sample 05: Minor Pavement Depression",
            }
            
            preset_choices = [k for k in preset_labels.keys() if k in available_samples]
            if not preset_choices:
                preset_choices = available_samples[:5]
                
            selected_choice = st.selectbox(
                "Select Road Survey Frame:",
                preset_choices,
                format_func=lambda x: preset_labels.get(x, x)
            )
            if selected_choice:
                selected_img_path = os.path.join(sample_dir, selected_choice)
                input_pil_img = Image.open(selected_img_path)
        else:
            uploaded_file = st.file_uploader("Upload road image (JPG/PNG format):", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                input_pil_img = Image.open(uploaded_file)

    with col_input2:
        if input_pil_img is not None:
            st.image(input_pil_img, caption="Loaded Survey Frame", width=420)

    # 2. AI Inference & Results Section (Placed BELOW image selection/upload)
    st.markdown("<hr style='margin: 18px 0; border: none; border-top: 1px solid rgba(128, 128, 128, 0.25);'>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">AI Inference & Quantification Analysis and its result</div>', unsafe_allow_html=True)
    
    if input_pil_img is not None:
        with st.spinner("Processing image through detection model and depth quantification..."):
            det_result = detector.detect(input_pil_img, conf_threshold=0.35, iou_threshold=0.45)
            
        col_out_img, col_out_stats = st.columns([1, 1])
        
        with col_out_img:
            st.image(det_result["annotated_image"], caption="Bounding Box Localization & Severity Classification", use_container_width=True)
            
        with col_out_stats:
            m1, m2, m3 = st.columns(3)
            m1.metric("Detected Instances", f"{det_result['total_potholes']}")
            m2.metric("Maximum Severity", f"{det_result['max_severity']} / 100")
            m3.metric("Average Severity", f"{det_result['avg_severity']} / 100")
            
            if det_result["total_potholes"] > 0:
                st.write("**Quantification Breakdown:**")
                df_det = pd.DataFrame([
                    {
                        "Instance": f"Pothole #{i+1}",
                        "Class": d["class_name"].capitalize(),
                        "Confidence": f"{d['confidence']*100:.1f}%",
                        "Surface Area": f"{d['area_cm2']} cm²",
                        "Est. Depth": f"{d['depth_cm']} cm",
                        "Severity Index": d["severity_score"],
                        "Category": "Critical" if d["severity_score"] >= 60 else ("High" if d["severity_score"] >= 40 else "Moderate")
                    }
                    for i, d in enumerate(det_result["detections"])
                ])
                st.dataframe(df_det, use_container_width=True)
            else:
                st.info("No road potholes localized. The surface is either within standard tolerance or confirmed as negative distress (asphalt patch / shadow).")
    else:
        st.write("Please select or upload a road frame above to view detection and quantification results.")

# =========================================================================
# TAB 2: DBSCAN GEOSPATIAL HOTSPOT MAP
# =========================================================================
with tab2:
    st.markdown('<div class="section-header">Geographical Hotspot Clustering (DBSCAN)</div>', unsafe_allow_html=True)
    
    col_db_ctrl, col_db_map = st.columns([1, 3])
    
    with col_db_ctrl:
        st.write("**Clustering Parameters**")
        eps_val = st.slider("Neighborhood Radius (Eps in meters):", min_value=100, max_value=800, value=350, step=25)
        min_pts = st.slider("Minimum Distress Points (MinPts):", min_value=2, max_value=8, value=3, step=1)
        
        pothole_records = telemetry_df[telemetry_df['true_potholes'] > 0].copy()
        pothole_records['severity_score'] = pothole_records['max_severity']
        
        clusterer = PotholeHotspotClusterer(eps_meters=float(eps_val), min_samples=min_pts)
        clustered_pts, summary_clusters, cluster_metrics = clusterer.cluster_points(pothole_records)
        
        st.markdown("<hr style='margin: 12px 0;'>", unsafe_allow_html=True)
        st.metric("Total Geotagged Points", f"{cluster_metrics['total_points']}")
        st.metric("Hotspot Clusters", f"{cluster_metrics['num_clusters']}")
        st.metric("Isolated / Noise Points", f"{cluster_metrics['num_noise_points']} ({cluster_metrics['noise_ratio']*100:.1f}%)")
        sil = cluster_metrics['silhouette_score']
        st.metric("Silhouette Coefficient", f"{sil if sil is not None else 'N/A'}")

    with col_db_map:
        if not clustered_pts.empty:
            mean_lat = clustered_pts['latitude'].mean()
            mean_lon = clustered_pts['longitude'].mean()
        else:
            mean_lat, mean_lon = 19.0760, 72.8777
            
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=13, tiles="OpenStreetMap")
        
        # Heatmap Layer
        heat_data = [[row['latitude'], row['longitude'], row['severity_score']] for _, row in clustered_pts.iterrows()]
        HeatMap(heat_data, radius=18, blur=14, max_zoom=14).add_to(m)
        
        cluster_colors = ['#dc2626', '#2563eb', '#16a34a', '#9333ea', '#d97706', '#0891b2', '#4f46e5']
        
        for _, c_row in summary_clusters.iterrows():
            cid = int(c_row['cluster_id'])
            c_color = cluster_colors[cid % len(cluster_colors)]
            
            folium.Circle(
                location=[c_row['centroid_latitude'], c_row['centroid_longitude']],
                radius=max(150, c_row['radius_meters']),
                color=c_color,
                fill=True,
                fill_color=c_color,
                fill_opacity=0.22,
                weight=2,
                tooltip=f"Hotspot #{cid} ({c_row['ward_name']})"
            ).add_to(m)
            
            popup_html = f"""
            <div style='font-family: Arial, sans-serif; font-size: 13px; width: 210px;'>
                <b style='color:#0f172a;'>Hotspot Cluster #{cid}</b><br>
                <b>Ward:</b> {c_row['ward_name']}<br>
                <b>Corridor:</b> {c_row['primary_road']}<br>
                <b>Pothole Count:</b> {c_row['pothole_count']}<br>
                <b>Mean Severity:</b> {c_row['average_severity']}/100<br>
                <b>Max Severity:</b> {c_row['max_severity']}/100<br>
                <b>Intensity Index:</b> {c_row['hotspot_intensity']}
            </div>
            """
            folium.Marker(
                location=[c_row['centroid_latitude'], c_row['centroid_longitude']],
                icon=folium.Icon(color="red", icon="info-sign"),
                popup=folium.Popup(popup_html, max_width=240)
            ).add_to(m)
            
        for _, pt in clustered_pts.iterrows():
            cid = pt['cluster_id']
            if cid == -1:
                folium.CircleMarker(
                    location=[pt['latitude'], pt['longitude']],
                    radius=4,
                    color="#64748b",
                    fill=True,
                    fill_color="#94a3b8",
                    fill_opacity=0.7,
                    tooltip=f"Isolated Distress (Severity: {pt['severity_score']})"
                ).add_to(m)
            else:
                c_color = cluster_colors[cid % len(cluster_colors)]
                folium.CircleMarker(
                    location=[pt['latitude'], pt['longitude']],
                    radius=5,
                    color=c_color,
                    fill=True,
                    fill_color=c_color,
                    fill_opacity=0.88,
                    tooltip=f"Cluster #{cid} | Severity: {pt['severity_score']}"
                ).add_to(m)
                
        st_folium(m, width="100%", height=480)
        
    st.write("**Hotspot Clusters Inventory**")
    if not summary_clusters.empty:
        st.dataframe(
            summary_clusters.rename(columns={
                "cluster_id": "Cluster ID",
                "ward_name": "Ward",
                "primary_road": "Primary Corridor",
                "pothole_count": "Distress Count",
                "average_severity": "Mean Severity",
                "max_severity": "Max Severity",
                "hotspot_intensity": "Intensity Score",
                "radius_meters": "Radius (m)"
            }),
            use_container_width=True
        )
