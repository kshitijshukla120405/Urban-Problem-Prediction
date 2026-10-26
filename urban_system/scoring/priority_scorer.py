"""
Multi-Criteria Risk and Priority Scoring Engine
Implements Section 6 priority formula:
  Priority = 0.35 * P + 0.25 * S + 0.20 * R + 0.10 * D + 0.10 * T
"""

class PriorityScorer:
    def __init__(self, weights=None):
        # Default weights from Section 6
        self.weights = weights or {
            "P": 0.35,  # Predicted Probability
            "S": 0.25,  # Severity Score
            "R": 0.20,  # Recurrence / Historical Frequency
            "D": 0.10,  # Population / Traffic Exposure
            "T": 0.10   # Recent Trend
        }
        
    def normalize_component(self, value, min_val, max_val):
        """Linearly scale value to 0 - 100 range."""
        if max_val <= min_val:
            return 50.0
        val = max(min_val, min(max_val, value))
        return round(((val - min_val) / (max_val - min_val)) * 100.0, 2)
        
    def calculate_priority(
        self,
        predicted_prob_pct,        # P: 0 to 100 (from XGBoost)
        severity_score,            # S: 0 to 100 (from YOLOv8 quantification)
        past_recurrence_count,     # Raw count (e.g. 0 to 10 past repairs/failures)
        traffic_pcu,               # Raw traffic PCU (e.g. 2000 to 80000)
        recent_trend_val           # Raw trend value (e.g. -1.0 to 3.0)
    ):
        """
        Calculates priority score using the validated municipal formula.
        Returns composite score (0-100), urgency category, and component breakdown.
        """
        P = max(0.0, min(100.0, float(predicted_prob_pct)))
        S = max(0.0, min(100.0, float(severity_score)))
        R = self.normalize_component(past_recurrence_count, min_val=0, max_val=10)
        D = self.normalize_component(traffic_pcu, min_val=2000, max_val=80000)
        T = self.normalize_component(recent_trend_val, min_val=-0.5, max_val=3.0)
        
        # Priority formula
        priority = (
            self.weights["P"] * P +
            self.weights["S"] * S +
            self.weights["R"] * R +
            self.weights["D"] * D +
            self.weights["T"] * T
        )
        priority = round(max(0.0, min(100.0, priority)), 2)
        
        # Categorization
        if priority >= 75.0:
            category = "Critical"
            sla_hours = 24
            badge_color = "#dc3545" # Red
        elif priority >= 55.0:
            category = "High"
            sla_hours = 72
            badge_color = "#fd7e14" # Orange
        elif priority >= 35.0:
            category = "Moderate"
            sla_hours = 168 # 7 days
            badge_color = "#ffc107" # Yellow
        else:
            category = "Low"
            sla_hours = 720 # 30 days
            badge_color = "#28a745" # Green
            
        return {
            "priority_score": priority,
            "category": category,
            "sla_hours": sla_hours,
            "badge_color": badge_color,
            "formula": "0.35*P + 0.25*S + 0.20*R + 0.10*D + 0.10*T",
            "components": {
                "P_predicted_prob": round(P, 1),
                "S_severity": round(S, 1),
                "R_recurrence_norm": round(R, 1),
                "D_exposure_norm": round(D, 1),
                "T_trend_norm": round(T, 1)
            },
            "weighted_contributions": {
                "P_contrib": round(self.weights["P"] * P, 2),
                "S_contrib": round(self.weights["S"] * S, 2),
                "R_contrib": round(self.weights["R"] * R, 2),
                "D_contrib": round(self.weights["D"] * D, 2),
                "T_contrib": round(self.weights["T"] * T, 2)
            }
        }

if __name__ == "__main__":
    scorer = PriorityScorer()
    res = scorer.calculate_priority(
        predicted_prob_pct=88.5,
        severity_score=72.0,
        past_recurrence_count=6,
        traffic_pcu=54000,
        recent_trend_val=2.1
    )
    print("Priority Calculation Test:")
    print(f"Priority Score: {res['priority_score']} / 100")
    print(f"Urgency Tier:   {res['category']} (SLA: {res['sla_hours']} hrs)")
    print("Weighted breakdown:", res["weighted_contributions"])
