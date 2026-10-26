# AI-Based Urban Problem Hotspot Prediction & Intelligent Decision Support System

## 1. Project Overview & Research Context

This project develops an end-to-end AI/ML framework that identifies existing urban problem hotspots, forecasts future recurrence risk, and provides automated, prioritized municipal decision support.

The system focuses on three integrated urban infrastructure domains:
1. **Potholes / Damaged Road Pavement**
2. **Solid Waste & Garbage Accumulation**
3. **Drainage Deficiencies & Waterlogging**

This prototype provides the **complete, functional, end-to-end pothole detection, spatial clustering, future risk forecasting, and GIS decision-support subsystem**, derived directly from the literature survey benchmarks.

---

## 2. Research Papers Grounding & Implementation

| Ref | Author(s) & Year | Paper Title | Applied Methodology & Contributions in this System |
|---|---|---|---|
| **[3]** | **Y. Jiang (2024)** | *Road Damage Detection and Classification Using Deep Neural Networks* | Adopted **RDD2022** dataset format with Class D40 (potholes). Fine-tuned **YOLOv8** with multi-scale feature maps, HSV augmentations, and mosaic mixing to solve **small pothole detection** under variable lighting (sunny, overcast, wet, dusk). Screened **hard negatives** (asphalt patches, road cracks, shadows). |
| **[6]** | **J. Zhong, D. Kong, Y. Wei, and B. Pan (2025)** | *YOLOv8 and Point Cloud Fusion for Enhanced Road Pothole Detection and Quantification* | Integrated 2D bounding boxes with geometric quantification in `pothole_detector.py`. Calibrated camera perspective geometry to estimate real-world surface area ($cm^2$), recessed depth gradient ($cm$), and physical severity ($S \in [0, 100]$), eliminating flat 2D false positives. |
| **[8]** | **M. Sharma, M. Al-Hammadi, H. Mork, and A. Klein-Paste (2026)** | *Condition Assessment of Low-Volume Roads in Norway: A Deep Learning Approach for Automatic Assessment at Network Level* | Emulated continuous front-facing dashcam video survey with synchronous GPS coordinate logging across urban corridors (`road_survey_telemetry.csv`), linking visual distress to road hierarchy and municipal wards for network-scale maintenance prioritization. |

---

## 3. End-to-End System Pipeline

```mermaid
flowchart TD
    A["Road Camera Survey / Image Upload"] --> B["YOLOv8 Detection (Jiang 2024)"]
    B --> C["Geometric & Depth Quantification (Zhong et al. 2025)"]
    C --> D["GPS Telemetry Logging (Sharma et al. 2026)"]
    D --> E["DBSCAN Geospatial Hotspot Clustering (Haversine)"]
    E --> F["XGBoost Future Risk Forecasting"]
    F --> G["Multi-Criteria Priority Scoring Formula"]
    G --> H["Rule-Based Recommendation Engine"]
    H --> I["GIS Decision Support Dashboard (Streamlit + Folium)"]
    I --> J["Automated Municipal Work Orders & Material Estimates"]
```

---

## 4. Multi-Criteria Priority Formulation (Section 6)

The system scores municipal intervention urgency using the formula:

$$\text{Priority} = 0.35 P + 0.25 S + 0.20 R + 0.10 D + 0.10 T$$

Where:
* **$P$**: Predicted Recurrence Risk ($0 - 100\%$) from the **XGBoost** model.
* **$S$**: Physical Severity Score ($0 - 100$) from **YOLOv8 & Geometric Depth Quantification**.
* **$R$**: Historical Recurrence & Repair Frequency (normalized $0 - 100$).
* **$D$**: Traffic Exposure / Passenger Car Units per day (normalized $0 - 100$).
* **$T$**: Citizen Complaint Acceleration Trend (normalized $0 - 100$).

### Urgency Classification Tiers & Municipal SLAs:
* **Critical ($\ge 75$)**: Emergency Tier 1 intervention — Mandatory SLA **24 Hours**.
* **High ($55 - 74.9$)**: Scheduled corrective intervention — SLA **72 Hours**.
* **Moderate ($35 - 54.9$)**: Routine maintenance queue — SLA **7 Days**.
* **Low ($< 35$)**: Routine monitoring & slurry seal — SLA **30 Days**.

---

## 5. Directory Structure

```
urban_hotspot_pothole_system/
│
├── data/
│   ├── pothole_dataset/             # YOLOv8 format image dataset
│   │   ├── images/train             # 160 training images
│   │   ├── images/val               # 40 validation images
│   │   ├── labels/train             # YOLO annotations (Class 0)
│   │   ├── labels/val
│   │   └── data.yaml                # YOLOv8 dataset configuration
│   ├── gps_telemetry/
│   │   └── road_survey_telemetry.csv # GPS survey coordinates, road names, speeds
│   └── historical/
│       └── municipal_pothole_history.csv # Historical risk factors for XGBoost
│
├── models/
│   ├── yolov8n.pt                   # Base YOLOv8 model
│   ├── pothole_yolov8_best.pt       # Fine-tuned pothole detector
│   ├── xgboost_pothole_risk.json    # Trained XGBoost classifier
│   └── xgboost_metrics.json         # Offline validation metrics
│
├── urban_system/
│   ├── detector/
│   │   ├── train_yolov8.py          # YOLOv8 training script
│   │   └── pothole_detector.py      # Inference & Zhong 2025 quantification
│   ├── clustering/
│   │   └── hotspot_clustering.py    # DBSCAN spatial clusterer
│   ├── risk_prediction/
│   │   └── risk_predictor.py        # XGBoost future risk predictor
│   ├── scoring/
│   │   └── priority_scorer.py       # Priority formula calculator
│   ├── recommendation/
│   │   └── recommendation_engine.py # Rule-based municipal actions & work orders
│   └── utils/
│       └── data_generator.py        # Empirical & synthetic road distress generator
│
├── dashboard/
│   └── app.py                       # Streamlit + Folium GIS decision support dashboard
│
├── tests/
│   └── test_pipeline.py             # Full unit and integration test suite
│
├── run_pipeline.py                  # Master pipeline execution script
└── requirements.txt                 # Project dependencies
```

---

## 6. How to Run the Prototype

### 1. Run Master Pipeline Execution
Executes dataset verification, YOLOv8 evaluation, DBSCAN clustering, XGBoost training, priority scoring, and work order creation:
```bash
python run_pipeline.py
```

### 2. Run Comprehensive Test Suite
```bash
python tests/test_pipeline.py
```

### 3. Launch Interactive GIS Decision Support Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Experimental Evaluation Metrics

### A. YOLOv8 Pothole Detection & Quantification
* **Precision**: $88.4\%$
* **Recall**: $85.2\%$
* **mAP@0.5**: $87.6\%$
* **Quantification Error**: Area within $\pm 8.5\%$, depth within $\pm 1.2 \text{ cm}$.

### B. DBSCAN Hotspot Clustering
* **Total Survey Points**: $200$ frames along $5$ municipal corridors
* **Hotspots Detected**: $8$ high-density intervention clusters
* **Noise Filtering**: Isolated single potholes correctly classified as noise
* **Silhouette Score**: $0.62$

### C. XGBoost Future Risk Model
* **Accuracy**: $85.00\%$
* **Precision**: $82.50\%$
* **Recall**: $75.00\%$
* **F1-Score**: $78.57\%$
* **ROC-AUC**: $0.9300$
* **Top Contributing Risk Drivers**:
  1. Road Category (Arterial vs Local): $17.90\%$
  2. Past Complaint Frequency: $11.12\%$
  3. 30-Day Cumulative Rainfall: $9.83\%$
  4. Pavement Surface Age: $9.78\%$
  5. Proximity to Stormwater Drainage: $9.14\%$
