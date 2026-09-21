import json
from pathlib import Path

with open("phase13_real_world_validation_summary.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"{'Image ID':<38} {'Ext':<4} {'Rec':<4} {'Miss':<5} {'Fuse':<5} {'Nutr':<10} {'Final Status':<40} {'Semantics':<10}")
print("-" * 120)

for r in data["results"]:
    img_id = r["image_id"]
    ext = r["extracted_count"]
    rec = r["recognized_count"]
    miss = r["missed_count"]
    fuse = r["fused_count"]
    nut = r.get("nutrition_status", "N/A")
    status = r["final_status"]
    sem = "OK" if r["semantics_ok"] else "FAIL"
    print(f"{img_id:<38} {ext:<4} {rec:<4} {miss:<5} {fuse:<5} {nut:<10} {status:<40} {sem:<10}")

print("-" * 120)
print(f"Total images: {data['total_images_tested']}")
print(f"Total time: {data['total_elapsed_seconds']}s")
print(f"Runtime: Min={data['runtime']['min_seconds']}s | Median={data['runtime']['median_seconds']}s | Max={data['runtime']['max_seconds']}s")
print(f"Peak memory: {data['memory']['peak_mb']} MB | Final memory: {data['memory']['final_mb']} MB")
