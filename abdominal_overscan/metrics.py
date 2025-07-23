# BLOCK 5 – Overscan metrics & summary stats

import pandas as pd

from .config import CSV_PATH

SUMMARY_CSV_PATH = CSV_PATH.with_name("summary_statistics.csv")

df = pd.read_csv(CSV_PATH, sep=None, engine="python", encoding="utf-8-sig")
df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

needed_cols = [
    "pubic_source", "caudal_overscan_mm", "cranial_overscan_mm",
    "scan_end_z_mm", "scan_start_z_mm"
]
for c in needed_cols:
    if c not in df.columns:
        df[c] = pd.NA

caudal_thresh = (
    df["pubic_source"].eq("FemurFallback").map({True: 50, False: 30})
    if "pubic_source" in df.columns else pd.Series(30, index=df.index)
)

df["caudal_overscan?"]  = (pd.to_numeric(df["caudal_overscan_mm"], errors="coerce")  > caudal_thresh).map({True: "yes", False: "no"})
df["cranial_overscan?"] =  pd.to_numeric(df["cranial_overscan_mm"],  errors="coerce").gt(30).map({True: "yes", False: "no"})
df["overscanning?"]     = ((df["caudal_overscan?"] == "yes") | (df["cranial_overscan?"] == "yes")).map({True: "yes", False: "no"})

df["calc_caudal_overscan_mm"]  = (pd.to_numeric(df["caudal_overscan_mm"],  errors="coerce") - caudal_thresh).clip(lower=0)
df["calc_cranial_overscan_mm"] = (pd.to_numeric(df["cranial_overscan_mm"], errors="coerce") - 30).clip(lower=0)
df["calc_total_overscan_mm"]   = df["calc_caudal_overscan_mm"] + df["calc_cranial_overscan_mm"]

scan_length = (pd.to_numeric(df["scan_end_z_mm"], errors="coerce") -
               pd.to_numeric(df["scan_start_z_mm"], errors="coerce")).replace(0, pd.NA)

percent_vals = (df["calc_total_overscan_mm"] / scan_length * 100).abs().round()
df["%_overscan"] = percent_vals.apply(lambda x: f"{int(x)}%" if pd.notna(x) else pd.NA)

with open(CSV_PATH, encoding="utf-8-sig") as fh:
    first_line = fh.readline()
sep = "\t" if "\t" in first_line else ","
df.to_csv(CSV_PATH, index=False, sep=sep, encoding="utf-8-sig")
print("Main CSV updated.")

caudal_excess  = df["calc_caudal_overscan_mm"]
cranial_excess = df["calc_cranial_overscan_mm"]
total_excess   = df["calc_total_overscan_mm"]

m_caudal = caudal_excess [caudal_excess  > 0].mean()
s_caudal = caudal_excess [caudal_excess  > 0].std()
m_cran   = cranial_excess[cranial_excess > 0].mean()
s_cran   = cranial_excess[cranial_excess > 0].std()
m_total  = total_excess  [total_excess   > 0].mean()
s_total  = total_excess  [total_excess   > 0].std()

fmt_mm  = lambda v: f"{int(round(v))} mm" if pd.notna(v) else "-"
fmt_pct = lambda v: f"{int(round(v))}%"  if pd.notna(v) else "-"

summary = pd.DataFrame({
    "METRIC": [
        "n",
        "Mean_caudal_overscan_excess",
        "SD_caudal_overscan_excess",
        "Mean_cranial_overscan_excess",
        "SD_cranial_overscan_excess",
        "Mean_total_overscan_excess",
        "SD_total_overscan_excess",
        "%_caudal_overscan",
        "%_cranial_overscan",
        "%_overscanning",
    ],
    "VALUE": [
        len(df),
        fmt_mm(m_caudal),
        fmt_mm(s_caudal),
        fmt_mm(m_cran),
        fmt_mm(s_cran),
        fmt_mm(m_total),
        fmt_mm(s_total),
        fmt_pct((df["caudal_overscan?"]  == "yes").mean() * 100),
        fmt_pct((df["cranial_overscan?"] == "yes").mean() * 100),
        fmt_pct((df["overscanning?"]    == "yes").mean() * 100),
    ],
})

summary.to_csv(SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig")
print(f"Summary statistics written → {SUMMARY_CSV_PATH}")
