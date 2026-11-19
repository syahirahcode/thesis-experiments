import pandas as pd
import json

# Baca file CSV
csv_path = "kb_rules - Mapping.csv"
df = pd.read_csv(csv_path)

kb_data = {}

def parse_list_field(val):
    if pd.isna(val):
        return []
    val_str = str(val).strip()
    try:
        return json.loads(val_str)
    except Exception:
        return [x.strip(' "[]\'') for x in val_str.split(',') if x.strip()]

for _, row in df.iterrows():
    season_code = str(row["Season Code"]).strip()

    kb_data[season_code] = {
        "season_code": season_code,
        "season_name": str(row["Season"]).strip(),
        "undertone": str(row["Under Tine"]).strip(),
        "target_skintone": parse_list_field(row["Skin tone"]),
        "target_saturation": parse_list_field(row["Skin Saturation"]),
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