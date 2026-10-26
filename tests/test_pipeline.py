"""
Unit & Integration Tests for the Urban Problem Hotspot Prediction System
"""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from urban_system.detector.pothole_detector import PotholeDetector
from urban_system.clustering.hotspot_clustering import PotholeHotspotClusterer
from urban_system.risk_prediction.risk_predictor import PotholeRiskPredictor
from urban_system.scoring.priority_scorer import PriorityScorer
from urban_system.recommendation.recommendation_engine import MunicipalRecommendationEngine

def test_data_integrity():
    gps_path = os.path.join(BASE_DIR, "data", "gps_telemetry", "road_survey_telemetry.csv")
    hist_path = os.path.join(BASE_DIR, "data", "historical", "municipal_pothole_history.csv")
    assert os.path.exists(gps_path), "GPS telemetry file missing"
    assert os.path.exists(hist_path), "Historical records file missing"
    
    df_gps = pd.read_csv(gps_path)
    df_hist = pd.read_csv(hist_path)
    assert len(df_gps) > 0, "GPS telemetry is empty"
    assert len(df_hist) > 0, "Historical data is empty"
    assert "latitude" in df_gps.columns and "longitude" in df_gps.columns
    assert "pothole_occurrence_label" in df_hist.columns

def test_pothole_detector_quantification():
    detector = PotholeDetector()
    test_img_path = os.path.join(BASE_DIR, "data", "pothole_dataset", "images", "val", "rdd_val_0000.jpg")
    assert os.path.exists(test_img_path)
    
    result = detector.detect(test_img_path, conf_threshold=0.25)
    assert "total_potholes" in result
    assert "detections" in result
    assert "annotated_image" in result
    assert isinstance(result["annotated_image"], np.ndarray)
    
    for det in result["detections"]:
        assert det["area_cm2"] >= 0
        assert det["depth_cm"] >= 0
        assert 0.0 <= det["severity_score"] <= 100.0

def test_dbscan_clustering():
    gps_path = os.path.join(BASE_DIR, "data", "gps_telemetry", "road_survey_telemetry.csv")
    df = pd.read_csv(gps_path)
    potholes = df[df["true_potholes"] > 0].copy()
    potholes["severity_score"] = potholes["max_severity"]
    
    clusterer = PotholeHotspotClusterer(eps_meters=350.0, min_samples=2)
    clustered_df, cluster_summary, metrics = clusterer.cluster_points(potholes)
    
    assert "cluster_id" in clustered_df.columns
    assert metrics["num_clusters"] > 0
    assert len(cluster_summary) == metrics["num_clusters"]
    for c in metrics["clusters"]:
        assert "centroid_latitude" in c
        assert "hotspot_intensity" in c

def test_xgboost_risk_prediction():
    hist_path = os.path.join(BASE_DIR, "data", "historical", "municipal_pothole_history.csv")
    predictor = PotholeRiskPredictor()
    metrics = predictor.train(hist_path)
    
    assert metrics["accuracy"] >= 0.70
    assert metrics["roc_auc"] >= 0.70
    
    test_road = {
        "road_category": "Arterial",
        "traffic_pcu": 40000,
        "heavy_vehicle_pct": 20.0,
        "pavement_age_years": 5.0,
        "drainage_proximity_m": 15.0,
        "rainfall_last_30d_mm": 200.0,
        "peak_rain_intensity_mmh": 35.0,
        "road_slope_pct": 2.0,
        "past_complaints_count": 5,
        "past_repairs_count": 2,
        "recent_complaint_trend": 1.0
    }
    pred = predictor.predict_risk(test_road)
    assert 0.0 <= pred["predicted_probability"] <= 100.0
    assert pred["risk_level"] in ["Low", "Moderate", "High", "Critical"]

def test_priority_scorer_formula():
    scorer = PriorityScorer()
    res = scorer.calculate_priority(
        predicted_prob_pct=80.0,
        severity_score=70.0,
        past_recurrence_count=5,
        traffic_pcu=50000,
        recent_trend_val=2.0
    )
    assert 0.0 <= res["priority_score"] <= 100.0
    assert res["category"] in ["Low", "Moderate", "High", "Critical"]
    assert res["sla_hours"] in [24, 72, 168, 720]
    
    # Check formula: 0.35*P + 0.25*S + 0.20*R + 0.10*D + 0.10*T
    expected = (
        0.35 * res["components"]["P_predicted_prob"] +
        0.25 * res["components"]["S_severity"] +
        0.20 * res["components"]["R_recurrence_norm"] +
        0.10 * res["components"]["D_exposure_norm"] +
        0.10 * res["components"]["T_trend_norm"]
    )
    assert abs(res["priority_score"] - round(expected, 2)) < 0.05

def test_recommendation_engine():
    engine = MunicipalRecommendationEngine()
    priority = {"category": "Critical", "priority_score": 85.0}
    hotspot = {"pothole_count": 4, "average_severity": 65.0, "max_severity": 80.0, "radius_m": 30.0}
    road = {"road_name": "Main Highway", "ward_name": "Ward 1", "road_category": "Arterial", "drainage_proximity_m": 8.0}
    
    wo = engine.generate_recommendation(priority, hotspot, road)
    assert "work_order_id" in wo
    assert wo["sla_hours"] == 24
    assert wo["pothole_metrics"]["required_asphalt_tonnes"] > 0
    assert wo["estimated_budget_inr"] > 0
    assert wo["drainage_correlation_action"] is not None

if __name__ == "__main__":
    print("Running pipeline tests...")
    test_data_integrity()
    print("✓ Data Integrity Passed")
    test_pothole_detector_quantification()
    print("✓ Pothole Detector Quantification Passed")
    test_dbscan_clustering()
    print("✓ DBSCAN Clustering Passed")
    test_xgboost_risk_prediction()
    print("✓ XGBoost Risk Prediction Passed")
    test_priority_scorer_formula()
    print("✓ Priority Scorer Formula Passed")
    test_recommendation_engine()
    print("✓ Recommendation Engine Passed")
    print("ALL TESTS PASSED SUCCESSFULLY!")
