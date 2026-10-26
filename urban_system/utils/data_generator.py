"""
Synthetic & Empirical Road Distress Data Generator
Inspired by:
  - Jiang (2024): Road Damage Detection & Classification using Deep Neural Networks (RDD2022 format)
  - Zhong et al. (2025): YOLOv8 Pothole Detection and Geometric Quantification
  - Sharma et al. (2026): Condition Assessment of Low-Volume Roads (GPS-referenced video survey)

Generates:
  1. YOLOv8 format image dataset (train/val) with potholes, small distresses, and hard negatives (patches, shadows, cracks).
  2. GPS-tagged survey telemetry CSV (latitude, longitude, speed, road segment, timestamp).
  3. Municipal historical records CSV for XGBoost future risk modeling.
"""

import os
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def create_road_surface(width=640, height=640, lighting="daylight"):
    """
    Synthesize realistic road asphalt surface with lane markings, aggregate texture, and lighting.
    """
    # Base asphalt color
    if lighting == "daylight":
        base_gray = random.randint(70, 95)
    elif lighting == "overcast":
        base_gray = random.randint(55, 75)
    elif lighting == "wet":
        base_gray = random.randint(40, 60)
    elif lighting == "dusk":
        base_gray = random.randint(50, 70)
    else:
        base_gray = 80

    # Asphalt noise matrix
    noise = np.random.normal(loc=base_gray, scale=12, size=(height, width)).clip(0, 255).astype(np.uint8)
    
    # Add aggregate fine pebble speckles
    fine_speckles = (np.random.rand(height, width) > 0.96).astype(np.uint8) * random.randint(30, 60)
    noise = np.clip(noise + fine_speckles, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(noise).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Perspective road lane markings
    # Lane line (dashed or solid)
    lane_color = (210, 210, 200) if lighting != "dusk" else (220, 200, 160)
    if lighting == "wet":
        lane_color = (230, 230, 230)
    
    # Draw roadside or lane line
    lane_x = random.choice([int(width * 0.18), int(width * 0.82), int(width * 0.5)])
    if random.random() > 0.2:
        # dashed lane line
        dash_length = 35
        gap = 25
        y = 0
        while y < height:
            draw.line([(lane_x, y), (lane_x, min(y + dash_length, height))], fill=lane_color, width=random.randint(4, 7))
            y += dash_length + gap

    # Apply lighting color tint
    if lighting == "wet":
        # Add reflective water sheen bands
        overlay = Image.new("RGB", (width, height), (30, 45, 60))
        img = Image.blend(img, overlay, alpha=0.15)
        # Add high-brightness reflection streaks
        refl_draw = ImageDraw.Draw(img)
        for _ in range(random.randint(2, 5)):
            rx = random.randint(50, width - 50)
            ry = random.randint(50, height - 50)
            rw = random.randint(40, 120)
            rh = random.randint(15, 40)
            refl_draw.ellipse([rx, ry, rx + rw, ry + rh], fill=(130, 140, 155, 60))
    elif lighting == "dusk":
        overlay = Image.new("RGB", (width, height), (90, 65, 45))
        img = Image.blend(img, overlay, alpha=0.20)

    return img

def inject_shadow(img):
    """Cast irregular diagonal tree / power pole shadow across the road."""
    width, height = img.size
    shadow_mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(shadow_mask)
    
    # Irregular polygon for tree branch shadow
    p1 = (random.randint(0, width // 2), 0)
    p2 = (random.randint(width // 2, width), 0)
    p3 = (random.randint(width // 2, width), height)
    p4 = (random.randint(0, width // 2), height)
    
    draw.polygon([p1, p2, p3, p4], fill=120)
    shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=random.randint(8, 20)))
    
    img_np = np.array(img).astype(np.float32)
    mask_np = np.array(shadow_mask).astype(np.float32) / 255.0
    
    # Darken region under shadow
    for c in range(3):
        img_np[:, :, c] = img_np[:, :, c] * (1.0 - 0.45 * mask_np)
        
    return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

def inject_road_patch(img):
    """Inject asphalt repair patch (hard negative sample - should NOT be detected as pothole)."""
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # Asphalt patch is rectangular or trapezoidal with sharp tar-sealed edges
    px = random.randint(80, width - 200)
    py = random.randint(80, height - 200)
    pw = random.randint(70, 180)
    ph = random.randint(60, 160)
    
    # Fresh dark asphalt or aged lighter patch
    patch_gray = random.choice([random.randint(25, 45), random.randint(110, 130)])
    patch_box = [px, py, px + pw, py + ph]
    draw.rectangle(patch_box, fill=(patch_gray, patch_gray, patch_gray))
    
    # Black bitumen seal line around the perimeter
    draw.rectangle(patch_box, outline=(20, 20, 20), width=random.randint(2, 4))
    return img

def inject_crack(img):
    """Inject longitudinal or alligator crack (hard negative sample for pure pothole detector)."""
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    cx = random.randint(60, width - 60)
    cy = random.randint(60, height - 150)
    
    num_segments = random.randint(8, 16)
    pts = [(cx, cy)]
    for _ in range(num_segments):
        cx += random.randint(-15, 18)
        cy += random.randint(10, 25)
        pts.append((cx, cy))
        
    for i in range(len(pts) - 1):
        draw.line([pts[i], pts[i+1]], fill=(25, 25, 25), width=random.randint(1, 3))
    return img

def inject_pothole(img, size_category="medium", is_wet=False):
    """
    Inject realistic pothole with fractured jagged boundaries, recessed depth gradient,
    loose aggregate debris, and optional water filling.
    Returns: modified image, bounding_box [xmin, ymin, xmax, ymax], severity_stats
    """
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # Determine dimensions based on size category (Jiang 2024 & Zhong 2025)
    if size_category == "small":
        # Small pothole (15-35 px on 640x640, ~10-25cm real life)
        rx = random.randint(10, 22)
        ry = random.randint(8, 18)
    elif size_category == "medium":
        # Medium pothole (35-70 px, ~25-50cm real life)
        rx = random.randint(25, 50)
        ry = random.randint(20, 42)
    else:  # large crater
        # Large pothole (70-130 px, >50cm real life)
        rx = random.randint(55, 95)
        ry = random.randint(45, 80)
        
    center_x = random.randint(rx + 30, width - rx - 30)
    center_y = random.randint(ry + 30, height - ry - 30)
    
    # Generate irregular jagged polygonal perimeter
    num_vertices = random.randint(16, 28)
    poly_pts = []
    for i in range(num_vertices):
        angle = 2.0 * math.pi * i / num_vertices
        # Add radial disturbance to simulate chipped asphalt edges
        r_jitter = random.uniform(0.78, 1.24)
        px = center_x + rx * math.cos(angle) * r_jitter
        py = center_y + ry * math.sin(angle) * r_jitter
        poly_pts.append((px, py))
        
    # Create mask for recessed depth shading
    mask_img = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    mask_draw.polygon(poly_pts, fill=255)
    
    # Draw dark base pit inside pothole
    pit_img = Image.new("RGB", (width, height), (0, 0, 0))
    pit_draw = ImageDraw.Draw(pit_img)
    
    if is_wet:
        # Muddy water / murky dark ponding
        pit_base_color = (25, 30, 35)
        pothole_inner = Image.new("RGB", (width, height), pit_base_color)
        inner_draw = ImageDraw.Draw(pothole_inner)
        # Specular water reflection streak
        inner_draw.ellipse([center_x - rx//3, center_y - ry//3, center_x + rx//3, center_y + ry//4], fill=(120, 130, 140))
    else:
        # Deep recessed pothole: center is darkest (depth), outer lip has coarse gravel
        pothole_inner = Image.new("RGB", (width, height), (22, 20, 18))
        inner_draw = ImageDraw.Draw(pothole_inner)
        # Inner broken sub-base gravel
        for _ in range(int(rx * ry * 0.05)):
            gx = center_x + random.uniform(-rx*0.7, rx*0.7)
            gy = center_y + random.uniform(-ry*0.7, ry*0.7)
            gcol = random.randint(45, 85)
            inner_draw.point((int(gx), int(gy)), fill=(gcol, gcol, gcol))
            
    # Composite pothole into road image using mask
    img.paste(pothole_inner, (0, 0), mask_img)
    
    # Draw chipped perimeter border (fractured asphalt lip)
    lip_color = (random.randint(110, 145), random.randint(110, 145), random.randint(110, 145))
    draw.polygon(poly_pts, outline=lip_color, width=2)
    
    # Calculate bounding box
    all_x = [p[0] for p in poly_pts]
    all_y = [p[1] for p in poly_pts]
    xmin = max(0, int(min(all_x)) - 3)
    ymin = max(0, int(min(all_y)) - 3)
    xmax = min(width - 1, int(max(all_x)) + 3)
    ymax = min(height - 1, int(max(all_y)) + 3)
    
    # Severity metrics (inspired by Zhong et al. 2025: area, perimeter, estimated depth cm)
    box_w_px = xmax - xmin
    box_h_px = ymax - ymin
    # Calibration factor: assume ~1.2 mm per pixel at standard dashcam geometry
    est_area_cm2 = round(box_w_px * box_h_px * 0.014, 1)
    est_depth_cm = round(random.uniform(3.5, 12.0) if size_category != "small" else random.uniform(2.0, 4.5), 1)
    
    severity_score = min(100.0, round((est_area_cm2 / 12.0) + (est_depth_cm * 5.5), 1))
    
    severity_stats = {
        "area_cm2": est_area_cm2,
        "depth_cm": est_depth_cm,
        "severity_score": severity_score,
        "is_wet": is_wet,
        "size_category": size_category
    }
    
    return img, [xmin, ymin, xmax, ymax], severity_stats

def generate_yolo_dataset(base_dir, num_train=180, num_val=45):
    """
    Generate complete YOLOv8 dataset with realistic images, labels, small potholes, and hard negatives.
    """
    train_img_dir = os.path.join(base_dir, "images", "train")
    train_lbl_dir = os.path.join(base_dir, "labels", "train")
    val_img_dir = os.path.join(base_dir, "images", "val")
    val_lbl_dir = os.path.join(base_dir, "labels", "val")
    
    splits = [
        ("train", num_train, train_img_dir, train_lbl_dir),
        ("val", num_val, val_img_dir, val_lbl_dir)
    ]
    
    catalog = []
    
    for split_name, count, img_dir, lbl_dir in splits:
        for idx in range(count):
            img_id = f"rdd_{split_name}_{idx:04d}"
            lighting = random.choice(["daylight", "daylight", "overcast", "wet", "dusk"])
            img = create_road_surface(640, 640, lighting=lighting)
            
            # 15% of images are hard negatives (no potholes, only patches, cracks, or heavy shadows)
            is_negative = (random.random() < 0.15)
            
            boxes = []
            severities = []
            
            if is_negative:
                neg_type = random.choice(["patch", "crack", "shadow", "patch_crack"])
                if "patch" in neg_type:
                    img = inject_road_patch(img)
                if "crack" in neg_type:
                    img = inject_crack(img)
                if "shadow" in neg_type:
                    img = inject_shadow(img)
            else:
                # 30% chance of shadow across road (lighting variation test)
                if random.random() < 0.30:
                    img = inject_shadow(img)
                # 20% chance of background crack or patch in addition to pothole
                if random.random() < 0.20:
                    img = inject_road_patch(img)
                if random.random() < 0.20:
                    img = inject_crack(img)
                    
                # Inject 1 to 3 potholes per image (including small, medium, large)
                num_potholes = random.choices([1, 2, 3], weights=[0.65, 0.25, 0.10])[0]
                for _ in range(num_potholes):
                    size_cat = random.choices(["small", "medium", "large"], weights=[0.35, 0.45, 0.20])[0]
                    wet = (lighting == "wet" or random.random() < 0.2)
                    img, box, stats = inject_pothole(img, size_category=size_cat, is_wet=wet)
                    boxes.append(box)
                    severities.append(stats)
            
            # Save image
            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            img.save(img_path, quality=92)
            
            # Save YOLO label format: class x_center y_center width height (normalized)
            lbl_path = os.path.join(lbl_dir, f"{img_id}.txt")
            with open(lbl_path, "w") as f:
                for box in boxes:
                    xmin, ymin, xmax, ymax = box
                    xc = ((xmin + xmax) / 2.0) / 640.0
                    yc = ((ymin + ymax) / 2.0) / 640.0
                    bw = (xmax - xmin) / 640.0
                    bh = (ymax - ymin) / 640.0
                    f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    
            catalog.append({
                "split": split_name,
                "image_id": img_id,
                "file_path": img_path,
                "label_path": lbl_path,
                "lighting": lighting,
                "pothole_count": len(boxes),
                "is_hard_negative": is_negative,
                "max_severity": max([s["severity_score"] for s in severities]) if severities else 0.0
            })
            
    print(f"Generated {len(catalog)} images across train and val splits.")
    return pd.DataFrame(catalog)

def generate_gps_telemetry(catalog_df, output_csv_path, city_center=(19.0760, 72.8777)):
    """
    Generate GPS survey telemetry mimicking mobile survey (Sharma et al. 2026).
    Coordinates simulated around real urban municipal road corridors.
    """
    base_lat, base_lon = city_center
    corridors = [
        {"name": "Western Express Highway", "ward": "Ward H-East", "pavement": "Asphalt Concrete", "speed_mean": 48},
        {"name": "SV Road Corridor", "ward": "Ward K-West", "pavement": "Bituminous Macadam", "speed_mean": 32},
        {"name": "Linking Road", "ward": "Ward H-West", "pavement": "Bituminous Surface", "speed_mean": 28},
        {"name": "LBS Marg Arterial", "ward": "Ward S", "pavement": "Dense Bituminous", "speed_mean": 35},
        {"name": "Industrial Link Rd", "ward": "Ward M-East", "pavement": "Flexible Pavement", "speed_mean": 24}
    ]
    
    rows = []
    # Create continuous sequential GPS traces along each corridor
    corridor_idx = 0
    cur_lat = base_lat
    cur_lon = base_lon
    
    for _, item in catalog_df.iterrows():
        corr = corridors[corridor_idx % len(corridors)]
        
        # Step forward along corridor
        step_lat = random.uniform(0.0004, 0.0012)
        step_lon = random.uniform(-0.0003, 0.0005)
        cur_lat += step_lat
        cur_lon += step_lon
        
        # Cluster switch
        if random.random() < 0.12:
            corridor_idx += 1
            cur_lat = base_lat + random.uniform(-0.05, 0.05)
            cur_lon = base_lon + random.uniform(-0.04, 0.04)
            
        speed = max(10, round(np.random.normal(corr["speed_mean"], 6), 1))
        
        rows.append({
            "image_id": item["image_id"],
            "split": item["split"],
            "file_name": os.path.basename(item["file_path"]),
            "timestamp": f"2026-10-25T{random.randint(8, 17):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}Z",
            "latitude": round(cur_lat, 6),
            "longitude": round(cur_lon, 6),
            "altitude_m": round(random.uniform(8.5, 32.0), 1),
            "vehicle_speed_kmh": speed,
            "road_name": corr["name"],
            "ward_name": corr["ward"],
            "pavement_type": corr["pavement"],
            "lighting_condition": item["lighting"],
            "true_potholes": item["pothole_count"],
            "max_severity": item["max_severity"],
            "is_hard_negative": item["is_hard_negative"]
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved GPS survey telemetry to {output_csv_path} with {len(df)} records.")
    return df

def generate_historical_risk_dataset(output_csv_path, num_records=600):
    """
    Generate ward and road-segment historical dataset for XGBoost risk prediction model.
    Features include spatial, environmental, drainage proximity, traffic, and recurrence factors.
    """
    wards = ["Ward H-East", "Ward H-West", "Ward K-West", "Ward S", "Ward M-East", "Ward F-North", "Ward G-South"]
    soil_types = ["Clayey Subgrade", "Sandy Loam", "Lateritic Soil", "Gravelly Silt"]
    pavement_types = ["Bituminous Macadam", "Dense Bituminous", "Asphalt Concrete", "Flexible Mastic"]
    
    data = []
    for i in range(num_records):
        seg_id = f"SEG_{i+1:04d}"
        ward = random.choice(wards)
        road_cat = random.choices(["Arterial", "Collector", "Local/Low-Volume"], weights=[0.30, 0.45, 0.25])[0]
        
        # Traffic volume in Passenger Car Units per day
        if road_cat == "Arterial":
            traffic_pcu = random.randint(25000, 75000)
            heavy_veh_pct = round(random.uniform(15.0, 32.0), 1)
        elif road_cat == "Collector":
            traffic_pcu = random.randint(10000, 30000)
            heavy_veh_pct = round(random.uniform(8.0, 18.0), 1)
        else:
            traffic_pcu = random.randint(2000, 12000)
            heavy_veh_pct = round(random.uniform(3.0, 10.0), 1)
            
        pavement_age_years = round(random.uniform(0.5, 12.0), 1)
        drainage_distance_m = round(random.uniform(5.0, 180.0), 1)
        subgrade_soil = random.choice(soil_types)
        pavement_type = random.choice(pavement_types)
        
        # Environmental / Weather
        rainfall_30d_mm = round(random.uniform(20.0, 480.0), 1)
        rain_intensity_peak = round(random.uniform(5.0, 75.0), 1)
        slope_pct = round(random.uniform(0.2, 8.5), 1)
        
        # Historical complaints & repair recurrence
        past_complaints_count = random.randint(0, 18)
        past_repairs_count = random.randint(0, 8)
        recent_complaint_trend = round(random.uniform(-0.5, 3.2), 2)  # positive means complaints accelerating
        
        # Pothole recurrence probability formula (latent ground truth)
        # Scaled to produce a balanced realistic 40/60 distribution across segments
        z_score = (
            - 2.8
            + 0.000035 * (traffic_pcu - 25000)
            + 0.05 * (heavy_veh_pct - 12.0)
            + 0.25 * (pavement_age_years - 5.0)
            + 0.006 * (rainfall_30d_mm - 200.0)
            + 0.03 * (rain_intensity_peak - 30.0)
            + 0.18 * (past_complaints_count - 6)
            + 0.28 * (past_repairs_count - 2)
            + 0.40 * recent_complaint_trend
            - 0.015 * (drainage_distance_m - 40.0)
        )
        prob = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, z_score))))
        binary_label = 1 if prob >= 0.50 else 0
        
        data.append({
            "segment_id": seg_id,
            "ward_name": ward,
            "road_category": road_cat,
            "traffic_pcu": traffic_pcu,
            "heavy_vehicle_pct": heavy_veh_pct,
            "pavement_age_years": pavement_age_years,
            "drainage_proximity_m": drainage_distance_m,
            "subgrade_soil": subgrade_soil,
            "pavement_type": pavement_type,
            "rainfall_last_30d_mm": rainfall_30d_mm,
            "peak_rain_intensity_mmh": rain_intensity_peak,
            "road_slope_pct": slope_pct,
            "past_complaints_count": past_complaints_count,
            "past_repairs_count": past_repairs_count,
            "recent_complaint_trend": recent_complaint_trend,
            "latent_risk_prob": round(prob, 4),
            "pothole_occurrence_label": binary_label
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved historical risk records to {output_csv_path} with {len(df)} records.")
    return df

if __name__ == "__main__":
    set_seed(42)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_dir = os.path.join(BASE_DIR, "data", "pothole_dataset")
    gps_path = os.path.join(BASE_DIR, "data", "gps_telemetry", "road_survey_telemetry.csv")
    hist_path = os.path.join(BASE_DIR, "data", "historical", "municipal_pothole_history.csv")
    
    print("Step 1: Generating YOLOv8 training and validation images & annotations...")
    catalog = generate_yolo_dataset(base_dir, num_train=160, num_val=40)
    
    print("\nStep 2: Generating GPS survey telemetry...")
    generate_gps_telemetry(catalog, gps_path)
    
    print("\nStep 3: Generating historical municipal risk dataset...")
    generate_historical_risk_dataset(hist_path, num_records=600)
    print("\nAll datasets generated successfully!")
