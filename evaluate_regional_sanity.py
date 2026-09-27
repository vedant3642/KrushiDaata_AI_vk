import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services.regional_stats_service import get_regional_stats_service
from train_torch import MultiTaskModel

def safe_index(lst, val):
    try:
        return lst.index(val)
    except ValueError:
        return 0

def run_evaluation():
    print("=" * 80)
    print("KRUSHIDAATA: EXTERNAL REGIONAL SANITY & GENERALIZABILITY EVALUATION")
    print("=" * 80)
    
    reg_service = get_regional_stats_service()
    
    # 1. Load PyTorch Model
    pt_path = Path("saved_models_2") / "multitask_model.pt"
    if not pt_path.exists():
        print(f"Error: {pt_path} not found.")
        return
        
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    pt_model = MultiTaskModel(
        num_districts=config["num_districts"],
        num_soils=config["num_soils"],
        num_numerical=config["num_numerical"],
        num_crops=config["num_crops"],
        num_fertilizers=config["num_fertilizers"]
    )
    pt_model.load_state_dict(checkpoint["model_state_dict"])
    pt_model.eval()
    
    districts = checkpoint["mappings"]["districts"]
    soils = checkpoint["mappings"]["soils"]
    crops = checkpoint["mappings"]["crops"]
    
    mean = np.array(checkpoint["scaler"]["mean"])
    scale = np.array(checkpoint["scaler"]["scale"])
    
    # 2. Load Real Soil Test Dataset
    df_soil = pd.read_csv("Crop and fertilizer dataset (1).csv")
    df_soil["District_Name"] = df_soil["District_Name"].str.strip()
    df_soil["Soil_color"] = df_soil["Soil_color"].str.strip()
    
    numerical_cols = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Rainfall", "Temperature"]
    
    results = []
    
    total_samples = 0
    aligned_gt_5 = 0
    aligned_gt_2 = 0
    low_volatility_count = 0
    
    district_breakdown = {}
    for d in districts:
        district_breakdown[d] = {
            "total": 0,
            "aligned_gt_5": 0,
            "aligned_gt_2": 0,
            "crops_recommended": {}
        }

    with torch.no_grad():
        for idx, row in df_soil.iterrows():
            dist_name = row["District_Name"]
            soil_name = row["Soil_color"]
            
            if dist_name not in districts or soil_name not in soils:
                continue
                
            dist_idx = districts.index(dist_name)
            soil_idx = soils.index(soil_name)
            
            raw_num = row[numerical_cols].values.astype(float)
            scaled_num = (raw_num - mean) / scale
            
            x_num = torch.FloatTensor(scaled_num).unsqueeze(0)
            x_dist = torch.LongTensor([dist_idx])
            x_soil = torch.LongTensor([soil_idx])
            
            crop_logits, _ = pt_model(x_num, x_dist, x_soil)
            _, pred_c = torch.max(crop_logits, 1)
            rec_crop = crops[pred_c.item()]
            
            # Query Regional Stats Service for historical area share & risk
            plausibility = reg_service.check_regional_plausibility(rec_crop, dist_name)
            risk = reg_service.get_yield_volatility_risk(rec_crop, dist_name)
            
            area_share = plausibility["area_share_pct"]
            cv_score = risk["cv_score"]
            
            total_samples += 1
            district_breakdown[dist_name]["total"] += 1
            district_breakdown[dist_name]["crops_recommended"][rec_crop] = district_breakdown[dist_name]["crops_recommended"].get(rec_crop, 0) + 1
            
            if area_share >= 5.0:
                aligned_gt_5 += 1
                district_breakdown[dist_name]["aligned_gt_5"] += 1
                
            if area_share >= 2.0:
                aligned_gt_2 += 1
                district_breakdown[dist_name]["aligned_gt_2"] += 1
                
            if cv_score <= 0.35:
                low_volatility_count += 1
                
    pct_gt_5 = (aligned_gt_5 / total_samples) * 100.0 if total_samples > 0 else 0
    pct_gt_2 = (aligned_gt_2 / total_samples) * 100.0 if total_samples > 0 else 0
    pct_stable = (low_volatility_count / total_samples) * 100.0 if total_samples > 0 else 0
    
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS: REGIONAL SANITY BENCHMARK")
    print("=" * 80)
    print(f"Total Soil Test Profiles Evaluated:               {total_samples}")
    print(f"Top-1 Crop Matching > 5% District Area Share:    {aligned_gt_5} / {total_samples} ({pct_gt_5:.2f}%)")
    print(f"Top-1 Crop Matching > 2% District Area Share:    {aligned_gt_2} / {total_samples} ({pct_gt_2:.2f}%)")
    print(f"Top-1 Crop Yield Volatility CV <= 0.35 (Stable):  {low_volatility_count} / {total_samples} ({pct_stable:.2f}%)")
    print("=" * 80)
    
    print("\nDISTRICT-WISE REGIONAL ALIGNMENT BREAKDOWN:")
    print("-" * 80)
    print(f"{'District':<15} | {'Samples':<8} | {'>5% Area Share':<18} | {'>2% Area Share':<18} | Top Crop Recommended")
    print("-" * 80)
    for dist_name, ddata in district_breakdown.items():
        n = ddata["total"]
        if n == 0:
            continue
        p5 = (ddata["aligned_gt_5"] / n) * 100.0
        p2 = (ddata["aligned_gt_2"] / n) * 100.0
        top_crop = max(ddata["crops_recommended"].items(), key=lambda x: x[1])[0] if ddata["crops_recommended"] else "N/A"
        print(f"{dist_name:<15} | {n:<8d} | {ddata['aligned_gt_5']:3d} ({p5:5.1f}%)       | {ddata['aligned_gt_2']:3d} ({p2:5.1f}%)       | {top_crop}")
    print("-" * 80)
    
    # Save results JSON
    summary_out = {
        "metric_name": "Regional Sanity & Historical Area Share Alignment Rate",
        "total_test_samples": total_samples,
        "regional_alignment_rate_gt_5_pct": round(pct_gt_5, 2),
        "regional_plausibility_rate_gt_2_pct": round(pct_gt_2, 2),
        "yield_stability_rate_cv_lt_35_pct": round(pct_stable, 2),
        "district_breakdown": district_breakdown
    }
    
    out_file = Path("evaluation_regional_sanity_results.json")
    with open(out_file, "w") as f:
        json.dump(summary_out, f, indent=4)
        
    print(f"\nEvaluation summary successfully saved to: {out_file.absolute()}")

if __name__ == "__main__":
    run_evaluation()
