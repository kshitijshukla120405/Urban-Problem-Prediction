"""
XGBoost Future Pothole Risk Prediction Engine
Trains on historical, spatial, temporal, and contextual municipal features to predict
future pothole recurrence probability P in [0, 100].
"""

import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PotholeRiskPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path or os.path.join(BASE_DIR, "models", "xgboost_pothole_risk.json")
        self.feature_names = [
            "traffic_pcu",
            "heavy_vehicle_pct",
            "pavement_age_years",
            "drainage_proximity_m",
            "rainfall_last_30d_mm",
            "peak_rain_intensity_mmh",
            "road_slope_pct",
            "past_complaints_count",
            "past_repairs_count",
            "recent_complaint_trend",
            "road_cat_Arterial",
            "road_cat_Collector",
            "road_cat_Local"
        ]
        self.model = None
        if os.path.exists(self.model_path):
            self.load_model(self.model_path)
            
    def preprocess_df(self, df):
        """Prepare feature matrix from raw municipal data."""
        df_processed = df.copy()
        
        # Categorical encoding for road category
        if "road_category" in df_processed.columns:
            df_processed["road_cat_Arterial"] = (df_processed["road_category"] == "Arterial").astype(int)
            df_processed["road_cat_Collector"] = (df_processed["road_category"] == "Collector").astype(int)
            df_processed["road_cat_Local"] = (df_processed["road_category"].str.startswith("Local")).astype(int)
        else:
            for col in ["road_cat_Arterial", "road_cat_Collector", "road_cat_Local"]:
                if col not in df_processed.columns:
                    df_processed[col] = 0
                    
        for col in self.feature_names:
            if col not in df_processed.columns:
                df_processed[col] = 0.0
                
        X = df_processed[self.feature_names].astype(float)
        return X

    def train(self, data_csv_path, test_size=0.20, random_state=42):
        """Train XGBoost model and compute full evaluation metrics."""
        print("=" * 60)
        print("AI URBAN HOTSPOT SYSTEM: TRAINING XGBOOST RISK PREDICTOR")
        print(f"Data source: {data_csv_path}")
        print("=" * 60)
        
        df = pd.read_csv(data_csv_path)
        X = self.preprocess_df(df)
        y = df["pothole_occurrence_label"].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.model = XGBClassifier(
            n_estimators=150,
            learning_rate=0.06,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_state
        )
        
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        
        # Feature importances
        importances = dict(zip(self.feature_names, [round(float(v), 4) for v in self.model.feature_importances_]))
        sorted_imp = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
        
        metrics = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc), 4),
            "feature_importance": sorted_imp
        }
        
        print("\n" + "=" * 50)
        print("XGBOOST POTHOLE RISK EVALUATION METRICS:")
        print(f"  Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision:       {prec:.4f} ({prec*100:.2f}%)")
        print(f"  Recall:          {rec:.4f} ({rec*100:.2f}%)")
        print(f"  F1-Score:        {f1:.4f} ({f1*100:.2f}%)")
        print(f"  ROC-AUC:         {roc:.4f}")
        print("=" * 50)
        print("\nTop 5 Risk Contributing Features:")
        for feat, imp in list(sorted_imp.items())[:5]:
            print(f"  - {feat}: {imp*100:.2f}%")
        print("=" * 50 + "\n")
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)
        
        # Save metrics json
        metrics_path = os.path.join(os.path.dirname(self.model_path), "xgboost_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        print(f"XGBoost model saved to {self.model_path}")
        return metrics

    def load_model(self, model_path):
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        self.model_path = model_path

    def predict_risk(self, feature_dict):
        """
        Predict probability of future pothole occurrence.
        Returns:
          probability: 0.0 to 100.0
          risk_level: Low, Moderate, High, Critical
          factors: contributions of top factors
        """
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded.")
            
        df_input = pd.DataFrame([feature_dict])
        X = self.preprocess_df(df_input)
        
        prob = float(self.model.predict_proba(X)[0, 1])
        prob_pct = round(prob * 100.0, 1)
        
        if prob_pct >= 75.0:
            level = "Critical"
        elif prob_pct >= 50.0:
            level = "High"
        elif prob_pct >= 25.0:
            level = "Moderate"
        else:
            level = "Low"
            
        return {
            "predicted_probability": prob_pct,
            "raw_probability": prob,
            "risk_level": level
        }

if __name__ == "__main__":
    csv_path = os.path.join(BASE_DIR, "data", "historical", "municipal_pothole_history.csv")
    predictor = PotholeRiskPredictor()
    metrics = predictor.train(csv_path)
    
    # Test sample prediction
    sample_road = {
        "road_category": "Arterial",
        "traffic_pcu": 45000,
        "heavy_vehicle_pct": 24.5,
        "pavement_age_years": 6.8,
        "drainage_proximity_m": 22.0,
        "rainfall_last_30d_mm": 280.0,
        "peak_rain_intensity_mmh": 45.0,
        "road_slope_pct": 1.2,
        "past_complaints_count": 8,
        "past_repairs_count": 4,
        "recent_complaint_trend": 1.8
    }
    pred = predictor.predict_risk(sample_road)
    print("\nSample Road Segment Risk Prediction:")
    print(f"Predicted Probability: {pred['predicted_probability']}%")
    print(f"Risk Level:            {pred['risk_level']}")
