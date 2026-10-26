"""
Intelligent Rule-Based Municipal Recommendation Engine
Transforms predicted risk, detected severity, and spatial cluster metrics into
actionable engineering interventions, resource allocations, and municipal work orders.
"""

import uuid
from datetime import datetime, timedelta

class MunicipalRecommendationEngine:
    def __init__(self):
        # Base unit costs (INR / USD standard municipal benchmarks)
        self.ASPHALT_TONNE_COST = 4500.0  # Cost per tonne of Hot Mix Asphalt
        self.CREW_DAILY_COST = 12000.0    # Daily crew labor & equipment cost
        
    def generate_recommendation(
        self,
        priority_info,
        hotspot_info,
        road_context
    ):
        """
        priority_info: output from PriorityScorer
        hotspot_info: dict with pothole_count, average_severity, max_severity, radius_m
        road_context: dict with road_name, ward_name, road_category, drainage_proximity_m
        """
        category = priority_info["category"]
        p_score = priority_info["priority_score"]
        pothole_count = hotspot_info.get("pothole_count", 1)
        avg_sev = hotspot_info.get("average_severity", 40.0)
        max_sev = hotspot_info.get("max_severity", 50.0)
        road_cat = road_context.get("road_category", "Collector")
        drainage_dist = road_context.get("drainage_proximity_m", 50.0)
        
        # Approximate damaged road area (m^2)
        # Zhong et al. (2025) quantification aggregation
        avg_area_m2 = (avg_sev * 0.08) * pothole_count
        avg_depth_m = min(0.12, (max_sev * 0.0018))
        
        # Volume with 1.25 compaction allowance
        compaction_factor = 1.25
        est_volume_m3 = round(avg_area_m2 * avg_depth_m * compaction_factor, 2)
        est_volume_m3 = max(0.15, est_volume_m3)
        # Bulk density of asphalt concrete ~ 2.4 tonnes/m3
        asphalt_tonnes = round(est_volume_m3 * 2.4, 2)
        
        # Rule-based decision logic
        if category == "Critical":
            urgency_text = "EMERGENCY TIER 1 - IMMEDIATE ACTION REQUIRED"
            sla_hours = 24
            if road_cat == "Arterial":
                intervention = "Deep Milling (50mm), Heavy Hot-Mix Asphalt Infill & High-Tack Bitumen Sealing"
                traffic_plan = "Temporary lane closure during night shift (11 PM - 5 AM); Traffic police escort advised."
            else:
                intervention = "Rapid Excavation, Base Compaction, and Hot-Pour Patch Resurfacing"
                traffic_plan = "Local detour warning signs; single-lane shuttle traffic control."
                
            equipment = [
                "1x 2.5T Vibratory Tandem Roller",
                "1x Asphalt Cutter & Jackhammer",
                "1x Bitumen Tack Coat Sprayer",
                "1x Hot-Mix Asphalt Thermal Dump Tipper",
                "Traffic Safety Cones & Illuminated Arrow Boards"
            ]
            crew_size = 6
            days_required = 1
            
        elif category == "High":
            urgency_text = "HIGH PRIORITY - SCHEDULED WITHIN 72 HOURS"
            sla_hours = 72
            intervention = "Infrared Pavement Heating, Crack Sealing & Polymer-Modified Cold/Warm Mix Overlay"
            traffic_plan = "Off-peak daytime coning with mobile flaggers."
            equipment = [
                "1x Light Walk-Behind Compactor",
                "1x Mechanical Bitumen Kettle",
                "1x Utility Service Truck",
                "Retroreflective Warning Signage"
            ]
            crew_size = 4
            days_required = 1
            
        elif category == "Moderate":
            urgency_text = "MODERATE PRIORITY - SCHEDULED WITHIN 7 DAYS"
            sla_hours = 168
            intervention = "Edge-Chipping, Aggregate Re-bedding, and Standard Cold-Mix Tamping"
            traffic_plan = "Routine shoulder / curbside work zone safety barriers."
            equipment = [
                "1x Hand Tamper / Plate Compactor",
                "1x Patch Repair Utility Van",
                "Cold-Mix Bagged Storage"
            ]
            crew_size = 3
            days_required = 1
            
        else: # Low
            urgency_text = "ROUTINE PREVENTIVE MONITORING - 30-DAY REVIEW"
            sla_hours = 720
            intervention = "Preventive Slurry Seal / Micro-surfacing & Routine Video Re-inspection"
            traffic_plan = "No lane closure required; continuous moving inspection."
            equipment = ["1x Mobile Dashcam Survey Unit"]
            crew_size = 2
            days_required = 0.5
            
        # Additional Drainage Recommendation (cross-problem synergy)
        drainage_action = None
        if drainage_dist < 15.0:
            drainage_action = "URGENT: Road damage is within 15m of stormwater drain. Inspect roadside catch basin and unblock silt to prevent recurring water accumulation."
        elif drainage_dist < 35.0:
            drainage_action = "NOTICE: Check curb side water runoff slope to prevent ponding on repaired pavement."
            
        # Cost estimate
        material_cost = asphalt_tonnes * self.ASPHALT_TONNE_COST
        labor_cost = crew_size * (self.CREW_DAILY_COST / 4) * days_required
        equipment_cost = 6000.0 if category in ["Critical", "High"] else 2000.0
        total_cost_est = round(material_cost + labor_cost + equipment_cost, 0)
        
        # Generate Work Order Identifier
        now = datetime.now()
        due_date = now + timedelta(hours=sla_hours)
        work_order_id = f"MUNI-WO-{now.strftime('%Y%m%d')}-{uuid.uuid4().hex[:5].upper()}"
        
        work_order = {
            "work_order_id": work_order_id,
            "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "sla_deadline": due_date.strftime("%Y-%m-%d %H:%M:%S"),
            "sla_hours": sla_hours,
            "priority_tier": category,
            "priority_score": p_score,
            "urgency_classification": urgency_text,
            "location": {
                "road_name": road_context.get("road_name", "Surveyed Road Segment"),
                "ward_name": road_context.get("ward_name", "Municipal Ward"),
                "road_category": road_cat,
                "coordinates": road_context.get("coordinates", [19.0760, 72.8777])
            },
            "pothole_metrics": {
                "cluster_potholes": pothole_count,
                "average_severity": avg_sev,
                "max_severity": max_sev,
                "estimated_area_m2": round(avg_area_m2, 2),
                "estimated_depth_cm": round(avg_depth_m * 100, 1),
                "required_asphalt_tonnes": asphalt_tonnes
            },
            "engineering_intervention": intervention,
            "drainage_correlation_action": drainage_action,
            "traffic_management": traffic_plan,
            "assigned_resources": {
                "crew_personnel": crew_size,
                "recommended_equipment": equipment,
                "estimated_duration_days": days_required
            },
            "estimated_budget_inr": total_cost_est
        }
        
        return work_order

if __name__ == "__main__":
    engine = MunicipalRecommendationEngine()
    sample_priority = {
        "category": "Critical",
        "priority_score": 82.4
    }
    sample_hotspot = {
        "pothole_count": 5,
        "average_severity": 68.5,
        "max_severity": 84.0,
        "radius_m": 45.0
    }
    sample_road = {
        "road_name": "Western Express Highway, Sector 4",
        "ward_name": "Ward H-East",
        "road_category": "Arterial",
        "drainage_proximity_m": 8.5,
        "coordinates": [19.0850, 72.8680]
    }
    wo = engine.generate_recommendation(sample_priority, sample_hotspot, sample_road)
    print("Generated Municipal Work Order:")
    for k, v in wo.items():
        print(f"  {k}: {v}")
