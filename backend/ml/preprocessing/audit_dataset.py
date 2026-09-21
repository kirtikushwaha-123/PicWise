import os
import re
from pathlib import Path
import pandas as pd
from difflib import SequenceMatcher

DEFAULT_FOOD_DATA_PATH = "data/food/ingredient_knowledge_base_500_with_alternate_names.csv"
DEFAULT_PERSONAL_CARE_DATA_PATH = "data/personal_care/personal_care_ingredients_dataset_csv.xlsx"

CLEANED_FOOD_PATH = "data/food/ingredient_knowledge_base_500_cleaned.csv"
CLEANED_PERSONAL_CARE_PATH = "data/personal_care/personal_care_ingredients_dataset_cleaned.xlsx"
REPORT_PATH = "backend/ml/preprocessing/audit_report.md"
CONFLICTS_CSV_PATH = "backend/ml/preprocessing/conflicting_alt_names.csv"

SAFETY_RANK = {"Very Safe": 1, "Safe": 2, "Moderate Risk": 3, "High Risk": 4, "Unknown": 0}
ALLERGY_RANK = {"None": 1, "Low": 2, "Medium": 3, "High": 4}


def normalize_safety_level(val):
    if pd.isna(val) or not str(val).strip():
        return "Unknown"
    s = str(val).strip().lower()
    if s in ["very safe"]:
        return "Very Safe"
    elif s in ["safe"]:
        return "Safe"
    elif s in ["moderate", "moderate risk"]:
        return "Moderate Risk"
    elif s in ["risky", "high risk"]:
        return "High Risk"
    return str(val).strip().title()


def normalize_allergy_risk(val):
    if pd.isna(val) or not str(val).strip():
        return "None"
    s = str(val).strip().lower()
    if s in ["none", "no risk", "n/a", "no"]:
        return "None"
    elif s in ["low"]:
        return "Low"
    elif s in ["medium", "moderate"]:
        return "Medium"
    elif s in ["high"]:
        return "High"
    return str(val).strip().title()


def clean_text_formatting(val):
    if pd.isna(val):
        return ""
    text = str(val)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_alt_names_string(val):
    if pd.isna(val):
        return ""
    text = str(val)
    parts = [re.sub(r"\s+", " ", p).strip() for p in text.split(";") if p.strip()]
    return "; ".join(parts)


def remove_alt_name_term(alt_str, term_to_remove):
    if pd.isna(alt_str) or not str(alt_str).strip():
        return ""
    terms = [t.strip() for t in str(alt_str).split(";") if t.strip()]
    filtered = [t for t in terms if t.lower() != term_to_remove.lower()]
    return "; ".join(filtered)


def apply_conflict_resolutions(food_df, pc_df):
    """
    Applies resolution rules for the 8 true conflicts to the food dataset.
    """
    if food_df.empty:
        return food_df, pc_df

    # 1. MERGE 1: Consolidate Soy Chunks, TVP, Soy Nuggets
    to_remove_tvp = [
        "Textured Vegetable Protein",
        "Soy Chunks (Textured Vegetable Protein)",
        "Soy Nuggets"
    ]
    food_df = food_df[~food_df["Ingredient Name"].str.strip().isin(to_remove_tvp)].copy()

    new_tvp_row = {
        "Ingredient Name": "Textured Vegetable Protein (Soy Chunks)",
        "Category": "Protein Sources",
        "Safety Level": "Safe",
        "Allergy Risk": "Medium",
        "Health Impact": "Positive",
        "Processing Level": "Minimally Processed",
        "Regulatory Status": "Approved",
        "Packaging Names / Alternate Names": "soya chunks; TVP; meal maker; soya nuggets; textured vegetable protein; soy chunks; soy nuggets; textured soy protein"
    }
    food_df = pd.concat([food_df, pd.DataFrame([new_tvp_row])], ignore_index=True)

    # 2. MERGE 2: Consolidate Riboflavin (Colour) & Vitamin B2 (Riboflavin)
    to_remove_riboflavin = [
        "Riboflavin (Colour)",
        "Vitamin B2 (Riboflavin)"
    ]
    food_df = food_df[~food_df["Ingredient Name"].str.strip().isin(to_remove_riboflavin)].copy()

    new_ribo_row = {
        "Ingredient Name": "Vitamin B2 (Riboflavin)",
        "Category": "Vitamins & Minerals",
        "Safety Level": "Very Safe",
        "Allergy Risk": "",
        "Health Impact": "Positive",
        "Processing Level": "Minimally Processed",
        "Regulatory Status": "Approved",
        "Packaging Names / Alternate Names": "riboflavin; vitamin B2; E101; INS 101; riboflavin color"
    }
    food_df = pd.concat([food_df, pd.DataFrame([new_ribo_row])], ignore_index=True)

    # 3. DISAMBIGUATE 1: Remove INS 322 from Sunflower Lecithin and Soy Lecithin
    # Add Lecithin (source unspecified)
    mask_sunflower = food_df["Ingredient Name"].str.strip() == "Sunflower Lecithin"
    mask_soy_lec = food_df["Ingredient Name"].str.strip() == "Soy Lecithin"

    if mask_sunflower.any():
        idx = food_df[mask_sunflower].index[0]
        curr_alt = food_df.loc[idx, "Packaging Names / Alternate Names"]
        food_df.loc[idx, "Packaging Names / Alternate Names"] = remove_alt_name_term(curr_alt, "INS 322")

    if mask_soy_lec.any():
        idx = food_df[mask_soy_lec].index[0]
        curr_alt = food_df.loc[idx, "Packaging Names / Alternate Names"]
        food_df.loc[idx, "Packaging Names / Alternate Names"] = remove_alt_name_term(curr_alt, "INS 322")

    new_lec_row = {
        "Ingredient Name": "Lecithin (source unspecified)",
        "Category": "Emulsifiers",
        "Safety Level": "Safe",
        "Allergy Risk": "Medium",
        "Health Impact": "Neutral",
        "Processing Level": "Processed",
        "Regulatory Status": "Approved",
        "Packaging Names / Alternate Names": "INS 322; lecithin (unspecified); lecithin"
    }
    food_df = pd.concat([food_df, pd.DataFrame([new_lec_row])], ignore_index=True)

    # 4. DISAMBIGUATE 2: Remove "vitamin E" from Vitamin E (Tocopheryl Acetate) & Tocopherols (Vitamin E)
    mask_vit_e1 = food_df["Ingredient Name"].str.strip() == "Vitamin E (Tocopheryl Acetate)"
    mask_vit_e2 = food_df["Ingredient Name"].str.strip() == "Tocopherols (Vitamin E)"

    if mask_vit_e1.any():
        idx = food_df[mask_vit_e1].index[0]
        curr_alt = food_df.loc[idx, "Packaging Names / Alternate Names"]
        food_df.loc[idx, "Packaging Names / Alternate Names"] = remove_alt_name_term(curr_alt, "vitamin E")

    if mask_vit_e2.any():
        idx = food_df[mask_vit_e2].index[0]
        curr_alt = food_df.loc[idx, "Packaging Names / Alternate Names"]
        food_df.loc[idx, "Packaging Names / Alternate Names"] = remove_alt_name_term(curr_alt, "vitamin E")

    return food_df, pc_df


def run_audit():
    food_path = Path(os.getenv("FOOD_DATA_PATH", DEFAULT_FOOD_DATA_PATH))
    pc_path = Path(os.getenv("PERSONAL_CARE_DATA_PATH", DEFAULT_PERSONAL_CARE_DATA_PATH))

    # Read raw datasets
    raw_food_df = pd.read_csv(food_path) if food_path.exists() else pd.DataFrame()
    if pc_path.exists():
        if pc_path.suffix.lower() == ".xlsx":
            raw_pc_df = pd.read_excel(pc_path)
        else:
            raw_pc_df = pd.read_csv(pc_path)
    else:
        raw_pc_df = pd.DataFrame()

    # Apply formatting clean & resolution rules
    food_df, pc_df = apply_conflict_resolutions(raw_food_df.copy(), raw_pc_df.copy())

    # Format text fields
    if not food_df.empty:
        food_df["Ingredient Name"] = food_df["Ingredient Name"].apply(clean_text_formatting)
        if "Packaging Names / Alternate Names" in food_df.columns:
            food_df["Packaging Names / Alternate Names"] = food_df["Packaging Names / Alternate Names"].apply(clean_alt_names_string)
        food_df.to_csv(CLEANED_FOOD_PATH, index=False)

    if not pc_df.empty:
        pc_df["Ingredient_Name"] = pc_df["Ingredient_Name"].apply(clean_text_formatting)
        if "Packaging Names / Alternate Names" in pc_df.columns:
            pc_df["Packaging Names / Alternate Names"] = pc_df["Packaging Names / Alternate Names"].apply(clean_alt_names_string)
        pc_df.to_excel(CLEANED_PERSONAL_CARE_PATH, index=False)

    report_lines = []
    report_lines.append("# PicWise Dataset Audit & Data Cleaning Report\n")
    report_lines.append(f"**Audit Timestamp**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # =========================================================================
    # 1. DUPLICATE CHECK
    # =========================================================================
    report_lines.append("## 1. Duplicate & Near-Duplicate Check\n")

    food_exact_dups = []
    if not food_df.empty:
        food_df["name_clean"] = food_df["Ingredient Name"].astype(str).apply(clean_text_formatting)
        counts = food_df["name_clean"].str.lower().value_counts()
        dup_names = counts[counts > 1].index.tolist()
        for d in dup_names:
            matches = food_df[food_df["name_clean"].str.lower() == d]
            food_exact_dups.append((d, len(matches)))

    pc_exact_dups = []
    if not pc_df.empty:
        pc_df["name_clean"] = pc_df["Ingredient_Name"].astype(str).apply(clean_text_formatting)
        counts = pc_df["name_clean"].str.lower().value_counts()
        dup_names = counts[counts > 1].index.tolist()
        for d in dup_names:
            matches = pc_df[pc_df["name_clean"].str.lower() == d]
            pc_exact_dups.append((d, len(matches)))

    report_lines.append(f"- **Food Dataset Exact Duplicates**: {len(food_exact_dups)} names found.")
    for name, count in food_exact_dups:
        report_lines.append(f"  - `{name}` (appears {count} times)")

    report_lines.append(f"- **Personal Care Dataset Exact Duplicates**: {len(pc_exact_dups)} names found.")
    for name, count in pc_exact_dups:
        report_lines.append(f"  - `{name}` (appears {count} times)")

    report_lines.append("\n### Candidate Near-Duplicate Merges (High Similarity / Shared Alt Names):\n")

    candidate_merges = []
    food_names = food_df["name_clean"].tolist() if not food_df.empty else []
    for i in range(len(food_names)):
        for j in range(i + 1, len(food_names)):
            n1, n2 = food_names[i], food_names[j]
            ratio = SequenceMatcher(None, n1.lower(), n2.lower()).ratio()
            if ratio >= 0.82 and n1.lower() != n2.lower():
                candidate_merges.append(("Food", n1, n2, round(ratio, 3)))

    pc_names = pc_df["name_clean"].tolist() if not pc_df.empty else []
    for i in range(len(pc_names)):
        for j in range(i + 1, len(pc_names)):
            n1, n2 = pc_names[i], pc_names[j]
            ratio = SequenceMatcher(None, n1.lower(), n2.lower()).ratio()
            if ratio >= 0.82 and n1.lower() != n2.lower():
                candidate_merges.append(("Personal Care", n1, n2, round(ratio, 3)))

    for domain, n1, n2, sim in candidate_merges[:30]:
        report_lines.append(f"- [{domain}] Candidate merge: `{n1}` <-> `{n2}` (similarity: {sim})")

    # =========================================================================
    # 2. ALTERNATE-NAME COLLISION CHECK
    # =========================================================================
    report_lines.append("\n## 2. Alternate-Name Collision Check\n")

    alt_index = {}

    if not food_df.empty:
        for idx, row in food_df.iterrows():
            c_name = row["name_clean"]
            safety = normalize_safety_level(row.get("Safety Level"))
            allergy = normalize_allergy_risk(row.get("Allergy Risk"))
            alt_raw = str(row.get("Packaging Names / Alternate Names", "") or "")

            if alt_raw and alt_raw.lower() != "nan":
                for alt in alt_raw.split(";"):
                    alt_c = clean_text_formatting(alt)
                    if alt_c:
                        key = alt_c.lower()
                        if key not in alt_index:
                            alt_index[key] = []
                        alt_index[key].append({
                            "dataset": "Food",
                            "alt_name": alt_c,
                            "canonical_name": c_name,
                            "safety_level": safety,
                            "allergy_risk": allergy
                        })

    if not pc_df.empty:
        alt_col = None
        for col in ["Packaging Names / Alternate Names", "Alternate_Names", "Synonyms"]:
            if col in pc_df.columns:
                alt_col = col
                break
        if alt_col:
            for idx, row in pc_df.iterrows():
                c_name = row["name_clean"]
                safety = normalize_safety_level(row.get("Safety_Level"))
                allergy = normalize_allergy_risk(row.get("Allergy_Risk"))
                alt_raw = str(row.get(alt_col, "") or "")
                if alt_raw and alt_raw.lower() != "nan":
                    for alt in alt_raw.split(";"):
                        alt_c = clean_text_formatting(alt)
                        if alt_c:
                            key = alt_c.lower()
                            if key not in alt_index:
                                alt_index[key] = []
                            alt_index[key].append({
                                "dataset": "Personal Care",
                                "alt_name": alt_c,
                                "canonical_name": c_name,
                                "safety_level": safety,
                                "allergy_risk": allergy
                            })

    bucket_a = []
    bucket_b = []

    for alt_key, occurrences in alt_index.items():
        if len(occurrences) > 1:
            canonical_set = set(o["canonical_name"] for o in occurrences)
            if len(canonical_set) > 1:
                safeties = set(o["safety_level"] for o in occurrences)
                allergies = set(o["allergy_risk"] for o in occurrences)

                if len(safeties) == 1 and len(allergies) == 1:
                    bucket_a.append({
                        "alt_name": occurrences[0]["alt_name"],
                        "canonical_names": list(canonical_set),
                        "safety_level": list(safeties)[0],
                        "allergy_risk": list(allergies)[0]
                    })
                else:
                    bucket_b.append({
                        "alt_name": occurrences[0]["alt_name"],
                        "datasets": list(set(o["dataset"] for o in occurrences)),
                        "canonical_names": list(canonical_set),
                        "safety_levels": list(safeties),
                        "allergy_risks": list(allergies)
                    })

    report_lines.append(f"- **Bucket (a) Harmless Collisions (Identical Labels)**: {len(bucket_a)} alternate names.")
    for item in bucket_a[:10]:
        report_lines.append(f"  - Alt Name: `{item['alt_name']}` -> Canonical: {item['canonical_names']} (Safety: {item['safety_level']}, Allergy: {item['allergy_risk']})")

    report_lines.append(f"\n- **Bucket (b) TRUE CONFLICTS (Differing Safety or Allergy Labels)**: **{len(bucket_b)}** alternate names remaining.")
    for item in bucket_b:
        report_lines.append(f"  - **Alt Name**: `{item['alt_name']}`")
        report_lines.append(f"    - Canonical Names: {', '.join(item['canonical_names'])}")
        report_lines.append(f"    - Conflicting Safety Levels: {item['safety_levels']}")
        report_lines.append(f"    - Conflicting Allergy Risks: {item['allergy_risks']}")

    conflicts_df_list = []
    for item in bucket_b:
        conflicts_df_list.append({
            "alt_name": item["alt_name"],
            "dataset_sources": "; ".join(item["datasets"]),
            "conflicting_canonical_names": "; ".join(item["canonical_names"]),
            "conflicting_safety_levels": "; ".join(item["safety_levels"]),
            "conflicting_allergy_risks": "; ".join(item["allergy_risks"])
        })
    conflicts_df = pd.DataFrame(conflicts_df_list)
    conflicts_df.to_csv(CONFLICTS_CSV_PATH, index=False)

    # =========================================================================
    # 3. MISSING / BLANK VALUE CHECK
    # =========================================================================
    report_lines.append("\n## 3. Missing / Blank Value Check\n")

    report_lines.append("### Food Dataset Missing Value Summary:\n")
    if not food_df.empty:
        for col in food_df.columns:
            if col in ["name_clean", "safety_norm", "allergy_norm"]:
                continue
            null_cnt = food_df[col].isna().sum() + (food_df[col].astype(str).str.strip() == "").sum()
            pct = (null_cnt / len(food_df)) * 100
            report_lines.append(f"- `{col}`: {null_cnt} missing / blank ({pct:.1f}%)")

    report_lines.append("\n### Personal Care Dataset Missing Value Summary:\n")
    if not pc_df.empty:
        for col in pc_df.columns:
            if col in ["name_clean", "safety_norm", "allergy_norm"]:
                continue
            null_cnt = pc_df[col].isna().sum() + (pc_df[col].astype(str).str.strip() == "").sum()
            pct = (null_cnt / len(pc_df)) * 100
            report_lines.append(f"- `{col}`: {null_cnt} missing / blank ({pct:.1f}%)")

    # =========================================================================
    # 4. LABEL CONSISTENCY BY CATEGORY & SPLIT OUTLIER GROUPS
    # =========================================================================
    report_lines.append("\n## 4. Label Consistency by Category & Outlier Detection\n")

    higher_risk_outliers = []
    lower_risk_outliers = []

    # Process Food
    if not food_df.empty:
        food_df["safety_norm"] = food_df["Safety Level"].apply(normalize_safety_level)
        food_df["allergy_norm"] = food_df["Allergy Risk"].apply(normalize_allergy_risk)

        for cat, group in food_df.groupby("Category"):
            if len(group) >= 4:
                # Safety Level outliers
                s_counts = group["safety_norm"].value_counts()
                top_s = s_counts.index[0]
                if s_counts.iloc[0] / len(group) >= 0.75:
                    for s_val, count in s_counts.items():
                        if count <= 2 and s_val != top_s:
                            outlier_rows = group[group["safety_norm"] == s_val]
                            for _, r in outlier_rows.iterrows():
                                item = {
                                    "dataset": "Food",
                                    "category": cat,
                                    "ingredient": r["name_clean"],
                                    "field": "Safety Level",
                                    "outlier_value": s_val,
                                    "dominant_value": top_s
                                }
                                if SAFETY_RANK.get(s_val, 0) > SAFETY_RANK.get(top_s, 0):
                                    higher_risk_outliers.append(item)
                                else:
                                    lower_risk_outliers.append(item)

                # Allergy Risk outliers
                a_counts = group["allergy_norm"].value_counts()
                top_a = a_counts.index[0]
                if a_counts.iloc[0] / len(group) >= 0.75:
                    for a_val, count in a_counts.items():
                        if count <= 2 and a_val != top_a:
                            outlier_rows = group[group["allergy_norm"] == a_val]
                            for _, r in outlier_rows.iterrows():
                                item = {
                                    "dataset": "Food",
                                    "category": cat,
                                    "ingredient": r["name_clean"],
                                    "field": "Allergy Risk",
                                    "outlier_value": a_val,
                                    "dominant_value": top_a
                                }
                                if ALLERGY_RANK.get(a_val, 0) > ALLERGY_RANK.get(top_a, 0):
                                    higher_risk_outliers.append(item)
                                else:
                                    lower_risk_outliers.append(item)

    # Process Personal Care
    if not pc_df.empty:
        pc_df["safety_norm"] = pc_df["Safety_Level"].apply(normalize_safety_level)
        pc_df["allergy_norm"] = pc_df["Allergy_Risk"].apply(normalize_allergy_risk)

        for cat, group in pc_df.groupby("Ingredient_Category"):
            if len(group) >= 4:
                # Safety Level
                s_counts = group["safety_norm"].value_counts()
                top_s = s_counts.index[0]
                if s_counts.iloc[0] / len(group) >= 0.75:
                    for s_val, count in s_counts.items():
                        if count <= 2 and s_val != top_s:
                            outlier_rows = group[group["safety_norm"] == s_val]
                            for _, r in outlier_rows.iterrows():
                                item = {
                                    "dataset": "Personal Care",
                                    "category": cat,
                                    "ingredient": r["name_clean"],
                                    "field": "Safety Level",
                                    "outlier_value": s_val,
                                    "dominant_value": top_s
                                }
                                if SAFETY_RANK.get(s_val, 0) > SAFETY_RANK.get(top_s, 0):
                                    higher_risk_outliers.append(item)
                                else:
                                    lower_risk_outliers.append(item)

                # Allergy Risk
                a_counts = group["allergy_norm"].value_counts()
                top_a = a_counts.index[0]
                if a_counts.iloc[0] / len(group) >= 0.75:
                    for a_val, count in a_counts.items():
                        if count <= 2 and a_val != top_a:
                            outlier_rows = group[group["allergy_norm"] == a_val]
                            for _, r in outlier_rows.iterrows():
                                item = {
                                    "dataset": "Personal Care",
                                    "category": cat,
                                    "ingredient": r["name_clean"],
                                    "field": "Allergy Risk",
                                    "outlier_value": a_val,
                                    "dominant_value": top_a
                                }
                                if ALLERGY_RANK.get(a_val, 0) > ALLERGY_RANK.get(top_a, 0):
                                    higher_risk_outliers.append(item)
                                else:
                                    lower_risk_outliers.append(item)

    report_lines.append(f"### Group 1: Outlier is HIGHER Risk than Category Peers ({len(higher_risk_outliers)} items)")
    report_lines.append("*Expected domain variations (e.g. allergens/additives in generally safe categories).*")
    for o in higher_risk_outliers:
        report_lines.append(f"- [{o['dataset']} | `{o['category']}`] `{o['ingredient']}` ({o['field']}): **{o['outlier_value']}** vs Category Dominant: **{o['dominant_value']}**")

    report_lines.append(f"\n### Group 2: Outlier is LOWER Risk than Category Peers ({len(lower_risk_outliers)} items)")
    report_lines.append("*Candidate rows for manual review (ingredient labeled safer than its category norm).*")
    for o in lower_risk_outliers:
        report_lines.append(f"- [{o['dataset']} | `{o['category']}`] `{o['ingredient']}` ({o['field']}): **{o['outlier_value']}** vs Category Dominant: **{o['dominant_value']}**")

    # =========================================================================
    # 5. SPELLING & FORMATTING NORMALIZATION
    # =========================================================================
    report_lines.append("\n## 5. Spelling & Formatting Normalization\n")
    report_lines.append(f"Applied resolution rules and non-semantic text formatting to `_cleaned` files.")
    report_lines.append(f"- Food dataset: `{CLEANED_FOOD_PATH}`")
    report_lines.append(f"- Personal care dataset: `{CLEANED_PERSONAL_CARE_PATH}`")

    # =========================================================================
    # 6. CLASS DISTRIBUTION SUMMARY
    # =========================================================================
    report_lines.append("\n## 6. Class Distribution Summary\n")

    report_lines.append("### Normalized Safety Level Distribution:")
    if not food_df.empty:
        report_lines.append("\n**Food Dataset Safety Level:**")
        for k, v in food_df["safety_norm"].value_counts().items():
            report_lines.append(f"- {k}: {v}")

    if not pc_df.empty:
        report_lines.append("\n**Personal Care Dataset Safety Level:**")
        for k, v in pc_df["safety_norm"].value_counts().items():
            report_lines.append(f"- {k}: {v}")

    comb_safety = pd.concat([food_df["safety_norm"], pc_df["safety_norm"]], ignore_index=True)
    report_lines.append("\n**Combined Safety Level:**")
    for k, v in comb_safety.value_counts().items():
        report_lines.append(f"- {k}: {v}")

    report_lines.append("\n### Normalized Allergy Risk Distribution:")
    if not food_df.empty:
        report_lines.append("\n**Food Dataset Allergy Risk:**")
        for k, v in food_df["allergy_norm"].value_counts().items():
            report_lines.append(f"- {k}: {v}")

    if not pc_df.empty:
        report_lines.append("\n**Personal Care Dataset Allergy Risk:**")
        for k, v in pc_df["allergy_norm"].value_counts().items():
            report_lines.append(f"- {k}: {v}")

    comb_allergy = pd.concat([food_df["allergy_norm"], pc_df["allergy_norm"]], ignore_index=True)
    report_lines.append("\n**Combined Allergy Risk:**")
    for k, v in comb_allergy.value_counts().items():
        report_lines.append(f"- {k}: {v}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Audit completed! True conflicts count: {len(bucket_b)}")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    run_audit()
