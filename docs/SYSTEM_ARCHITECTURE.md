# System Architecture & Academic Presentation Guide

## AI-Based Urban Problem Hotspot Prediction & Intelligent Decision Support System

---

## 1. Executive Summary for Project Evaluation & PPT

This prototype operationalizes the research recommendations from three landmark publications:

1. **Y. Jiang (2024)**: Solves the *small pothole* and *lighting variation* problems via high-resolution, scale-invariant feature extraction and hard-negative screening on the RDD2022 dataset structure.
2. **J. Zhong et al. (2025)**: Solves the *2D false-positive problem* (shadows, oil stains, flat bitumen patches) by integrating 2D YOLOv8 bounding boxes with morphological and depth-gradient quantification to compute true physical volume ($cm^3$) and severity ($S \in [0, 100]$).
3. **M. Sharma et al. (2026)**: Establishes the *network-scale road condition assessment framework* using forward-facing camera feeds coupled with synchronous GPS coordinate logging to enable municipality-wide condition indexing without cost-prohibitive laser profiling equipment.

---

## 2. Mathematical Formulations

### A. DBSCAN Geospatial Neighborhood (Haversine Distance)
Given two GPS points on Earth surface $p_1 = (\phi_1, \lambda_1)$ and $p_2 = (\phi_2, \lambda_2)$ in radians:
$$d(p_1, p_2) = 2 R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
Where $R = 6,371,000 \text{ m}$.
Points $p$ with $|N_\epsilon(p)| \ge MinPts$ are designated as **Core Hotspots**; remaining points are classified as borderline or **Noise (isolated incidents)**.

### B. Zhong et al. Physical Severity Metric
Given bounding box dimensions $(w, h)$ in pixels and normalized vertical coordinate $\hat{y} \in [0, 1]$:
$$\text{Pixel Scale: } \mu(\hat{y}) = 0.8 + 1.4(1.0 - \hat{y}) \quad (\text{mm/px})$$
$$\text{Area: } A_{\text{est}} = \left(\frac{w \cdot \mu}{10}\right) \times \left(\frac{h \cdot \mu}{10}\right) \times 0.78 \quad (\text{cm}^2)$$
$$\text{Depth: } D_{\text{est}} = \max\left(1.5, \min\left(14.0, \frac{\Delta I_{\text{contrast}}}{8.0} + \frac{A_{\text{est}}}{250.0}\right)\right) \quad (\text{cm})$$
$$\text{Severity Score: } S = \min\left(50, \frac{A_{\text{est}}}{1200} \cdot 50\right) + \min\left(50, \frac{D_{\text{est}}}{12} \cdot 50\right) \in [0, 100]$$

### C. Multi-Criteria Priority Scoring Formula (Section 6)
$$\text{Priority} = 0.35 P + 0.25 S + 0.20 R + 0.10 D + 0.10 T$$
* $P \in [0, 100]$: XGBoost predicted probability of recurrence.
* $S \in [0, 100]$: Physical severity from YOLOv8 & depth quantification.
* $R \in [0, 100]$: Normalized historical repair recurrence count.
* $D \in [0, 100]$: Traffic exposure (Passenger Car Units / day).
* $T \in [0, 100]$: Citizen complaint acceleration trend.

---

## 3. Recommended PPT Slide Outline

| Slide # | Slide Title | Key Content & Visuals |
|---|---|---|
| **1** | Title Slide | Project Title, Candidate Names, Department, Guide Name |
| **2** | Motivation & Problem Statement | Reactive municipal complaint handling vs Proactive AI-driven maintenance. 3 Urban problems: Potholes, Waste, Waterlogging. |
| **3** | Literature Survey & Research Gaps | Summary table comparing Jiang (2024), Zhong (2025), Sharma (2026). Identification of false positives, lighting variance, and lack of unified decision support. |
| **4** | Proposed Unified System Architecture | End-to-end pipeline diagram: Vision $\rightarrow$ GPS $\rightarrow$ DBSCAN $\rightarrow$ XGBoost $\rightarrow$ Priority Scoring $\rightarrow$ Recommendation $\rightarrow$ GIS Dashboard. |
| **5** | Stage 1: Computer Vision & 3D Quantification | YOLOv8 fine-tuned on RDD2022 format. Show visual comparisons of small potholes, wet pavement reflections, and screening of hard negatives (patches/shadows). |
| **6** | Stage 2: DBSCAN Geospatial Clustering | Haversine distance spatial grouping. Distinguishing core municipal hotspot clusters from isolated noise points. Show Folium map screenshot. |
| **7** | Stage 3: XGBoost Future Risk Modeling | Feature importance bar chart (Arterial category, 30-day rain, drainage distance, pavement age). ROC-AUC = 0.930. |
| **8** | Stage 4: Multi-Criteria Priority Formulation | Formula breakdown ($0.35P + 0.25S + 0.20R + 0.10D + 0.10T$). Threshold classification: Critical ($\ge 75$), High, Moderate, Low. |
| **9** | Stage 5: Rule-Based Recommendation Engine | Automated Work Order generation: Engineering intervention, crew sizing, equipment, estimated asphalt tonnage, and budget calculation. |
| **10** | Interactive GIS Decision Support Dashboard | Screenshots of live Streamlit + OpenStreetMap application showcasing multi-tier inspection views. |
| **11** | Experimental Results & Verification | Precision, Recall, mAP@0.5, ROC-AUC, Silhouette Score tables. |
| **12** | Conclusion & Future Extensions | Expansion to Solid Waste (DBSCAN + XGBoost) and Drainage Waterlogging (Elevation + Rainfall intensity integration). |
