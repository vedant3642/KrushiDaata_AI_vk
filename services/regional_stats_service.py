import os
import pandas as pd
import numpy as np
from pathlib import Path

SOIL_TO_YIELD_CROP_MAP = {
    "Cotton": "Cotton(Lint)",
    "Ginger": "Dry Ginger",
    "Gram": "Gram",
    "Grapes": "Grapes",
    "Groundnut": "Groundnut",
    "Jowar": "Jowar",
    "Maize": "Maize",
    "Masoor": "Masoor",
    "Moong": "Moong(Green Gram)",
    "Rice": "Rice",
    "Soybean": "Soyabean",
    "Sugarcane": "Sugarcane",
    "Tur": "Arhar/Tur",
    "Turmeric": "Turmeric",
    "Urad": "Urad",
    "Wheat": "Wheat"
}

YIELD_TO_SOIL_CROP_MAP = {v: k for k, v in SOIL_TO_YIELD_CROP_MAP.items()}

CROP_TYPES_MAP = {
    "Cotton": "Fiber Crops",
    "Ginger": "Spices",
    "Gram": "Pulses",
    "Grapes": "Fruits",
    "Groundnut": "Oilseeds",
    "Jowar": "Cereals",
    "Maize": "Cereals",
    "Masoor": "Pulses",
    "Moong": "Pulses",
    "Rice": "Cereals",
    "Soybean": "Oilseeds",
    "Sugarcane": "Sugar / Cash Crops",
    "Tur": "Pulses",
    "Turmeric": "Spices",
    "Urad": "Pulses",
    "Wheat": "Cereals"
}

ESTIMATED_MARKET_PRICES = {
    "Cotton": 62000,
    "Ginger": 45000,
    "Gram": 54000,
    "Grapes": 50000,
    "Groundnut": 60000,
    "Jowar": 30000,
    "Maize": 22000,
    "Masoor": 60000,
    "Moong": 72000,
    "Rice": 23000,
    "Soybean": 46000,
    "Sugarcane": 3500,
    "Tur": 70000,
    "Turmeric": 85000,
    "Urad": 69000,
    "Wheat": 22750
}

ESTIMATED_CULTIVATION_COST_PER_HA = {
    "Cotton": 45000,
    "Ginger": 80000,
    "Gram": 25000,
    "Grapes": 120000,
    "Groundnut": 35000,
    "Jowar": 20000,
    "Maize": 28000,
    "Masoor": 22000,
    "Moong": 22000,
    "Rice": 40000,
    "Soybean": 30000,
    "Sugarcane": 110000,
    "Tur": 26000,
    "Turmeric": 75000,
    "Urad": 22000,
    "Wheat": 32000
}

class RegionalStatsService:
    def __init__(self, csv_path="crop-wise-area-production-yield.csv"):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {self.csv_path.absolute()}")
            
        print(f"Loading regional stats dataset from '{self.csv_path}'...")
        self.df = pd.read_csv(self.csv_path)
        self._preprocess()

    def _preprocess(self):
        # Clean string columns
        self.df["crop_name_clean"] = self.df["crop_name"].astype(str).str.strip()
        self.df["district_name_clean"] = self.df["district_name"].astype(str).str.strip()
        self.df["state_name_clean"] = self.df["state_name"].astype(str).str.strip()
        self.df["season_clean"] = self.df["season"].astype(str).str.strip()
        
        # Clean numeric fields
        self.df["area"] = pd.to_numeric(self.df["area"], errors="coerce").fillna(0.0)
        self.df["yield"] = pd.to_numeric(self.df["yield"], errors="coerce").fillna(0.0)
        self.df["production"] = pd.to_numeric(self.df["production"], errors="coerce").fillna(0.0)
        
        # Exclude 'Total' season for area share calculation
        self.df_seasonal = self.df[~self.df["season_clean"].isin(["Total"])].copy()
        
        # Precompute district total cropped area
        self.district_total_area = self.df_seasonal.groupby("district_name_clean")["area"].sum().to_dict()
        
        # Fast O(1) Precomputed lookup dictionary for (district, crop) area sum
        self.dist_crop_area = self.df_seasonal.groupby(["district_name_clean", "crop_name_clean"])["area"].sum().to_dict()
        
        # Fast O(1) Precomputed lookup for yield statistics
        yield_group = self.df[self.df["yield"] > 0].groupby(["district_name_clean", "crop_name_clean"])["yield"]
        self.yield_means = yield_group.mean().to_dict()
        self.yield_stds = yield_group.std().to_dict()
        self.yield_recent_means = yield_group.apply(lambda x: x.tail(5).mean()).to_dict()
        self.yield_counts = yield_group.count().to_dict()
        
    def get_mapped_crop_name(self, soil_crop_name):
        return SOIL_TO_YIELD_CROP_MAP.get(soil_crop_name.strip(), soil_crop_name.strip())

    def get_expected_yield(self, crop_name, district_name, season="All", last_n_years=5):
        yield_crop = self.get_mapped_crop_name(crop_name)
        dist_clean = district_name.strip()
        key = (dist_clean, yield_crop)
        
        if key in self.yield_recent_means and not np.isnan(self.yield_recent_means[key]):
            avg_yield_ha = float(self.yield_recent_means[key])
            source = f"District ({dist_clean})"
        elif key in self.yield_means and not np.isnan(self.yield_means[key]):
            avg_yield_ha = float(self.yield_means[key])
            source = f"District Average ({dist_clean})"
        else:
            avg_yield_ha = 1.5
            source = "Default Benchmark"
            
        yield_quintal_acre = avg_yield_ha * 4.047
        
        return {
            "crop": crop_name,
            "district": dist_clean,
            "yield_tonne_per_ha": round(avg_yield_ha, 3),
            "yield_quintal_per_acre": round(yield_quintal_acre, 2),
            "data_source": source
        }

    def get_yield_volatility_risk(self, crop_name, district_name):
        yield_crop = self.get_mapped_crop_name(crop_name)
        dist_clean = district_name.strip()
        key = (dist_clean, yield_crop)
        
        cnt = self.yield_counts.get(key, 0)
        mean_y = self.yield_means.get(key, 1.5)
        std_y = self.yield_stds.get(key, 0.5)
        
        if cnt >= 3 and mean_y > 0 and not np.isnan(std_y):
            cv = std_y / (mean_y + 1e-6)
        else:
            cv = 0.35
            std_y = 0.5
            mean_y = 1.5
            
        if cv < 0.25:
            risk_tier = "Low Risk (High Stability)"
            risk_badge = "🟢 Low Risk"
            risk_score = 1
        elif cv <= 0.45:
            risk_tier = "Moderate Risk"
            risk_badge = "🟡 Moderate Risk"
            risk_score = 2
        else:
            risk_tier = "High Volatility Risk"
            risk_badge = "🔴 High Risk"
            risk_score = 3
            
        return {
            "crop": crop_name,
            "district": dist_clean,
            "cv_score": round(float(cv), 3),
            "std_yield": round(float(std_y), 3),
            "mean_yield": round(float(mean_y), 3),
            "risk_tier": risk_tier,
            "risk_badge": risk_badge,
            "risk_score": risk_score
        }

    def check_regional_plausibility(self, crop_name, district_name, area_threshold_pct=2.0):
        yield_crop = self.get_mapped_crop_name(crop_name)
        dist_clean = district_name.strip()
        
        total_dist_area = self.district_total_area.get(dist_clean, 1.0)
        crop_area = self.dist_crop_area.get((dist_clean, yield_crop), 0.0)
        
        area_share_pct = (crop_area / (total_dist_area + 1e-6)) * 100.0
        is_plausible = area_share_pct >= area_threshold_pct
        
        if is_plausible:
            badge = f"✅ Regionally Established ({area_share_pct:.1f}% area share)"
            warning = None
        else:
            if crop_area == 0:
                badge = "⚠️ Regionally Uncommon (0% historical area share)"
                warning = f"This crop is agronomically favorable based on soil, but has not been historically cultivated in {dist_clean}."
            else:
                badge = f"⚠️ Regionally Uncommon (<{area_threshold_pct}% area share)"
                warning = f"This crop is agronomically favorable based on soil, but historically uncommon in {dist_clean} (only {area_share_pct:.2f}% of local cropped area)."
                
        return {
            "crop": crop_name,
            "district": dist_clean,
            "is_plausible": is_plausible,
            "area_share_pct": round(float(area_share_pct), 2),
            "cultivated_area_ha": round(float(crop_area), 1),
            "total_district_area_ha": round(float(total_dist_area), 1),
            "plausibility_badge": badge,
            "warning_message": warning
        }

    def get_profit_estimate(self, crop_name, district_name, custom_price_per_ton=None):
        yield_info = self.get_expected_yield(crop_name, district_name)
        yield_ha = yield_info["yield_tonne_per_ha"]
        
        price_per_ton = custom_price_per_ton or ESTIMATED_MARKET_PRICES.get(crop_name, 30000)
        cost_per_ha = ESTIMATED_CULTIVATION_COST_PER_HA.get(crop_name, 30000)
        
        gross_revenue_ha = yield_ha * price_per_ton
        net_profit_ha = gross_revenue_ha - cost_per_ha
        net_profit_acre = net_profit_ha / 2.47105
        
        return {
            "crop": crop_name,
            "district": district_name,
            "yield_tonne_ha": yield_ha,
            "market_price_per_ton": price_per_ton,
            "gross_revenue_ha": round(gross_revenue_ha, 2),
            "cost_per_ha": round(cost_per_ha, 2),
            "net_profit_ha": round(net_profit_ha, 2),
            "net_profit_acre": round(net_profit_acre, 2)
        }

    def get_top_district_crops(self, district_name, top_n=5):
        dist_clean = district_name.strip()
        total_dist_area = self.district_total_area.get(dist_clean, 1.0)
        
        # Get all crops in district from precomputed map
        dist_crops = [(crop, area) for (d, crop), area in self.dist_crop_area.items() if d == dist_clean]
        dist_crops.sort(key=lambda x: x[1], reverse=True)
        
        top_crops = []
        for y_name, area in dist_crops[:top_n]:
            soil_name = YIELD_TO_SOIL_CROP_MAP.get(y_name, y_name)
            share_pct = (area / (total_dist_area + 1e-6)) * 100.0
            avg_y = self.yield_recent_means.get((dist_clean, y_name), 1.5)
            top_crops.append({
                "crop": soil_name,
                "yield_crop_name": y_name,
                "area_share_pct": round(float(share_pct), 2),
                "avg_yield_ha": round(float(avg_y), 2),
                "crop_type": CROP_TYPES_MAP.get(soil_name, "Other")
            })
            
        return top_crops

    def get_crops_by_type(self, crop_type_filter="All"):
        all_crops = list(SOIL_TO_YIELD_CROP_MAP.keys())
        if crop_type_filter == "All":
            return all_crops
        return [c for c in all_crops if CROP_TYPES_MAP.get(c) == crop_type_filter]

_regional_stats_instance = None

def get_regional_stats_service(csv_path="crop-wise-area-production-yield.csv"):
    global _regional_stats_instance
    if _regional_stats_instance is None:
        _regional_stats_instance = RegionalStatsService(csv_path=csv_path)
    return _regional_stats_instance
