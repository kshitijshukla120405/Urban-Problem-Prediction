"""
End-to-End Master Pipeline Runner
Executes the entire research workflow:
  1. Data Verification & Generation
  2. YOLOv8 Pothole Detection Training & Evaluation
  3. DBSCAN Spatial Hotspot Clustering
  4. XGBoost Future Risk Model Training & Evaluation
  5. Multi-Criteria Priority Scoring
  6. Municipal Recommendation & Work Order Generation
"""

import os
import sys
import json
import pandas as pd

# Set base dir dynamically to support any computer / clone location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from urban_system.utils.data_generator import generate_yolo_dataset, generate_gps_telemetry, generate_historical_risk_dataset, set_seed
from urban_system.detector.train_yolov8 import train_pothole_detector
from urban_system.detector.pothole_detector import PotholeDetector
from urban_system.clustering.hotspot_clustering import PotholeHotspotClusterer
from urban_system.risk_prediction.risk_predictor import PotholeRiskPredictor
from urban_system.scoring.priority_scorer import PriorityScorer
from urban_system.recommendation.recommendation_engine import MunicipalRecommendationEngine

def run_complete_pipeline(train_yolo_epochs=5):
    print("\n" + "=" * 75)
    print("AI-BASED URBAN PROBLEM HOTSPOT PREDICTION & DECISION SUPPORT SYSTEM")
    print("MASTER WORKFLOW EXECUTION")
    print("=" * 75)
    
    # Paths
    dataset_yaml = os.path.join(BASE_DIR, "data", "pothole_dataset", "data.yaml")
    gps_path = os.path.join(BASE_DIR, "data", "gps_telemetry", "road_survey_telemetry.csv")
    hist_path = os.path.join(BASE_DIR, "data", "historical", "municipal_pothole_history.csv")
    model_dir = os.path.join(BASE_DIR, "models")
    best_weights = os.path.join(model_dir, "pothole_yolov8_best.pt")
    
    # -------------------------------------------------------------
    # STEP 1: DATASET VERIFICATION / GENERATION
    # -------------------------------------------------------------
    print("\n>>> [STEP 1/6] Verifying Urban Distress Datasets...")
    if not os.path.exists(gps_path) or not os.path.exists(hist_path):
        print("Generating synthetic and empirical research datasets...")
        set_seed(42)
        catalog = generate_yolo_dataset(os.path.join(BASE_DIR, "data", "pothole_dataset"), num_train=160, num_val=40)
        generate_gps_telemetry(catalog, gps_path)
        generate_historical_risk_dataset(hist_path, num_records=600)
    else:
        print(f"Datasets verified at {gps_path} and {hist_path}")
        
    # -------------------------------------------------------------
    # STEP 2: YOLOv8 POTHOLE DETECTION TRAINING / LOADING
    # -------------------------------------------------------------
    print("\n>>> [STEP 2/6] YOLOv8 Pothole Detector Training & Evaluation...")
    if not os.path.exists(best_weights):
        print(f"Training YOLOv8 on pothole dataset for {train_yolo_epochs} epochs...")
        yolo_metrics = train_pothole_detector(
            data_yaml_path=dataset_yaml,
            output_model_dir=model_dir,
            epochs=train_yolo_epochs,
            batch_size=8
        )
    else:
        print(f"Trained YOLOv8 model found at {best_weights}.")
        yolo_metrics = {"status": "Model ready", "model_path": best_weights}

    # Test detection & quantification on a sample validation image
    detector = PotholeDetector(best_weights if os.path.exists(best_weights) else None)
    test_img = os.path.join(BASE_DIR, "data", "pothole_dataset", "images", "val", "rdd_val_0000.jpg")
    det_sample = detector.detect(test_img)
    print(f"Test Frame Analysis ({os.path.basename(test_img)}):")
    print(f"  - Detected Potholes: {det_sample['total_potholes']}")
    print(f"  - Max Severity Score: {det_sample['max_severity']}/100")
    for idx, d in enumerate(det_sample["detections"]):
        print(f"    * #{idx+1}: Area={d['area_cm2']}cm², Depth={d['depth_cm']}cm, Sev={d['severity_score']}, Conf={d['confidence']*100:.1f}%")

    # -------------------------------------------------------------
    # STEP 3: DBSCAN GEOSPATIAL HOTSPOT CLUSTERING
    # -------------------------------------------------------------
    print("\n>>> [STEP 3/6] DBSCAN Geospatial Hotspot Clustering (GPS Telemetry)...")
    telemetry = pd.read_csv(gps_path)
    pothole_pts = telemetry[telemetry['true_potholes'] > 0].copy()
    pothole_pts['severity_score'] = pothole_pts['max_severity']
    
    clusterer = PotholeHotspotClusterer(eps_meters=350.0, min_samples=3)
    clustered_df, cluster_summary, dbscan_metrics = clusterer.cluster_points(pothole_pts)
    
    print(f"  - Total GPS Pothole Coordinates: {dbscan_metrics['total_points']}")
    print(f"  - Hotspot Clusters Identified:    {dbscan_metrics['num_clusters']}")
    print(f"  - Isolated Incidents (Noise):     {dbscan_metrics['num_noise_points']} ({dbscan_metrics['noise_ratio']*100:.1f}%)")
    print(f"  - Silhouette Score:               {dbscan_metrics['silhouette_score']}")

    # -------------------------------------------------------------
    # STEP 4: XGBOOST FUTURE RISK MODEL TRAINING
    # -------------------------------------------------------------
    print("\n>>> [STEP 4/6] XGBoost Future Risk Model Training...")
    risk_predictor = PotholeRiskPredictor()
    xgb_metrics = risk_predictor.train(hist_path)
    
    # -------------------------------------------------------------
    # STEP 5: PRIORITY SCORING (Section 6 Formula)
    # -------------------------------------------------------------
    print("\n>>> [STEP 5/6] Multi-Criteria Priority Scoring...")
    scorer = PriorityScorer()
    
    # Evaluate top hotspot cluster
    top_cluster = cluster_summary.sort_values(by="hotspot_intensity", ascending=False).iloc[0]
    
    # Predict risk for the road corridor hosting top cluster
    corridor_features = {
        "road_category": "Arterial",
        "traffic_pcu": 52000,
        "heavy_vehicle_pct": 24.0,
        "pavement_age_years": 7.2,
        "drainage_proximity_m": 10.5,
        "rainfall_last_30d_mm": 310.0,
        "peak_rain_intensity_mmh": 50.0,
        "road_slope_pct": 1.5,
        "past_complaints_count": 9,
        "past_repairs_count": 5,
        "recent_complaint_trend": 2.2
    }
    risk_out = risk_predictor.predict_risk(corridor_features)
    predicted_prob = risk_out["predicted_probability"]
    
    priority_result = scorer.calculate_priority(
        predicted_prob_pct=predicted_prob,
        severity_score=top_cluster["average_severity"],
        past_recurrence_count=5,
        traffic_pcu=52000,
        recent_trend_val=2.2
    )
    
    print(f"Top Hotspot: Cluster #{int(top_cluster['cluster_id'])} on {top_cluster['primary_road']} ({top_cluster['ward_name']})")
    print(f"  - Potholes in Cluster: {top_cluster['pothole_count']}")
    print(f"  - Average Severity (S): {top_cluster['average_severity']}")
    print(f"  - Predicted Risk Probability (P): {predicted_prob}%")
    print(f"  - Final Priority Score: {priority_result['priority_score']} / 100")
    print(f"  - Urgency Tier: {priority_result['category']} (Mandatory SLA: {priority_result['sla_hours']} hrs)")

    # -------------------------------------------------------------
    # STEP 6: MUNICIPAL RECOMMENDATION & WORK ORDER GENERATION
    # -------------------------------------------------------------
    print("\n>>> [STEP 6/6] Rule-Based Recommendation Engine & Work Order...")
    rec_engine = MunicipalRecommendationEngine()
    hotspot_ctx = {
        "pothole_count": int(top_cluster["pothole_count"]),
        "average_severity": top_cluster["average_severity"],
        "max_severity": top_cluster["max_severity"],
        "radius_m": top_cluster["radius_meters"]
    }
    road_ctx = {
        "road_name": top_cluster["primary_road"],
        "ward_name": top_cluster["ward_name"],
        "road_category": "Arterial",
        "drainage_proximity_m": 10.5,
        "coordinates": [top_cluster["centroid_latitude"], top_cluster["centroid_longitude"]]
    }
    work_order = rec_engine.generate_recommendation(priority_result, hotspot_ctx, road_ctx)
    
    print(f"Generated Municipal Work Order: {work_order['work_order_id']}")
    print(f"  - Engineering Intervention: {work_order['engineering_intervention']}")
    if work_order['drainage_correlation_action']:
        print(f"  - Cross-Problem Action: {work_order['drainage_correlation_action']}")
    print(f"  - Required Asphalt: {work_order['pothole_metrics']['required_asphalt_tonnes']} Tonnes")
    print(f"  - Assigned Personnel: {work_order['assigned_resources']['crew_personnel']} workers")
    print(f"  - Estimated Budget: ₹{work_order['estimated_budget_inr']:,.0f}")
    
    print("\n" + "=" * 75)
    print("ALL WORKFLOW STAGES EXECUTED SUCCESSFULLY!")
    print("=" * 75 + "\n")
    
    return {
        "yolo": yolo_metrics,
        "dbscan": dbscan_metrics,
        "xgboost": xgb_metrics,
        "priority": priority_result,
        "work_order": work_order
    }

if __name__ == "__main__":
    run_complete_pipeline(train_yolo_epochs=5)
