import json

import pandas as pd

# Baca file CSV (ekspor tab Mapping dari workbook CFY)
csv_path = "kb_rules - Mapping.csv"
df = pd.read_csv(csv_path)

# Workbook memakai header "Undertone"; versi lama memakai "Under Tine".
UNDERTONE_COL = "Undertone" if "Undertone" in df.columns else "Under Tine"
COMBINATION_COL = "Kombinasi Atribut"

kb_data = {}


def parse_list_field(val):
    if pd.isna(val):
        return []
    val_str = str(val).strip()
    try:
        return [str(x).strip() for x in json.loads(val_str)]
    except Exception:
        return [x.strip(' "[]\'') for x in val_str.split(',') if x.strip()]


def parse_combinations(val):
    """Kolom Kombinasi Atribut: [["Skin tone", "Saturation"], ...]."""
    if pd.isna(val) or not str(val).strip():
        return []
    pairs = json.loads(str(val).strip())
    return [[str(t).strip(), str(s).strip()] for t, s in pairs]


for _, row in df.iterrows():
    season_code = str(row["Season Code"]).strip()
    combinations = (
        parse_combinations(row[COMBINATION_COL]) if COMBINATION_COL in df.columns else []
    )

    kb_data[season_code] = {
        "season_code": season_code,
        "season_name": str(row["Season"]).strip(),
        "undertone": str(row[UNDERTONE_COL]).strip(),
        "target_skintone": parse_list_field(row["Skin tone"]),
        "target_saturation": parse_list_field(row["Skin Saturation"]),
        # Kombinasi skin tone x saturation yang sah untuk season ini (Mapping v4 CFY).
        # Jika ada, pencocokan season WAJIB memakai field ini, bukan perkalian
        # target_skintone x target_saturation.
        "target_combinations": combinations,
        "makeup_look": str(row["Makeup Look"]).strip(),
        "look_file_name": str(row["Look File Name"]).strip(),
        "eye_makeup": {
            "technique": str(row["Eye Look"]).strip(),
            "direction": str(row["Eye Color Direction"]).strip(),
            "hex_colors": parse_list_field(row["Eye Hex Color"])
        },
        "blush_makeup": {
            "application": str(row["Blush Look"]).strip(),
            "direction": str(row["Blush Color Direction"]).strip(),
            "hex_colors": parse_list_field(row["Blush Hex Color"])
        },
        "lip_makeup": {
            "finish": str(row["Lip Look"]).strip(),
            "direction": str(row["Lip Color Direction"]).strip(),
            "hex_colors": parse_list_field(row["Lip Hex Color"])
        },
        "complexion_finish": str(row["Complexion"]).strip()
    }

# Simpan ke file kb_rules.json
output_path = "app/core/kb_rules.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(kb_data, f, indent=2, ensure_ascii=False)

print(f"Berhasil mengonversi {len(kb_data)} season ke {output_path}")