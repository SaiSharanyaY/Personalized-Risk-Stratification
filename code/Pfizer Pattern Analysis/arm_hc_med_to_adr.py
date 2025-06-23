
import pandas as pd
import numpy as np
import os
import re
from scipy.stats import gamma
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
INPUT_FILE = "Data/Final datasets/pfizer_with_labels_dose_1+2.csv"
TARGET_COLUMNS = [
    'Gastrointestinal Issues', 'Pain Syndromes', 'Psychological Disorders',
    'Musculoskeletal Disorders', 'Fever', 'Dermatological Conditions',
    'Neurological Disorders', 'Postural Disorders', 'Cardiovascular Conditions', 'Respiratory Symptoms', 'Injection Site Reaction'
]
MIN_SUPPORT = 0.0005
MIN_CONFIDENCE = 0.2
MIN_LIFT = 1
OUTPUT_FOLDER = "Code/Pfizer Pattern Analysis/ARM_HC_Med_to_ADR"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# preprocessing functions
def clean_text_item(item):
    item = re.sub(r"[^\w\s\-']", '', item)
    return item.strip().lower()

def calc_ic_ebgm(obs, exp):
    ic = np.log2((obs + 0.5) / (exp + 0.5))
    var = (1/(obs + 0.5) + 1/(exp + 0.5)) / (np.log(2)**2)
    ic_sd = np.sqrt(var)
    ic_lower = ic - 2 * ic_sd
    a = obs + 1
    rate = exp + 1
    ebgm = a / rate
    eb05 = gamma.ppf(0.05, a, scale=1/rate)
    return ic, ic_sd, ic_lower, ebgm, eb05

def build_transactions(row):
    
    items = []
    if pd.notnull(row["ExtractedMedications"]):
        items += [f"{clean_text_item(m)} [medication]" for m in row["ExtractedMedications"].split(',') if m.strip()]
    if pd.notnull(row["ExtractedPastHC"]):
        items += [f"{clean_text_item(hc)} [health condition]" for hc in eval(row["ExtractedPastHC"]) if isinstance(hc, str) and hc.strip()]

    return items

# === MAIN ===
df = pd.read_csv(INPUT_FILE, low_memory=False)

for col in TARGET_COLUMNS:
    print(f"🔍 Processing: {col}")
    df_target = df[df[col].notnull()].copy()
    transactions = []
    for _, row in df_target.iterrows():
        items = build_transactions(row)
        adrs = [f"{s.strip().lower()} [{col.lower()}]" for s in str(row[col]).split(',') if s.strip()]
        if items and adrs:
            transactions.append(items + adrs)
    if not transactions:
        print("⚠️ Skipped - no valid transactions")
        continue

    te = TransactionEncoder()
    df_encoded = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)

    # Mine rules
    freq_items = fpgrowth(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules = rules[
        rules['antecedents'].apply(lambda x: any('[medication]' in i for i in x) and any('[health condition]' in i for i in x)) &
        rules['consequents'].apply(lambda x: all(f"[{col.lower()}]" in i for i in x)) &
        (rules['lift'] >= MIN_LIFT)
    ].copy()

    if rules.empty:
        print("⚠️ No strong rules found.")
        continue

    rules["lhs"] = rules["antecedents"].apply(lambda x: list(x)[0])
    rules["rhs"] = rules["consequents"].apply(lambda x: list(x)[0])
    rules = rules.drop_duplicates(subset=["lhs", "rhs"])

    # Add IC/EBGM
    enriched = []
    for _, r in rules.iterrows():
        lhs, rhs = [r["lhs"]], [r["rhs"]]
        mask_lhs = df_encoded[lhs].all(axis=1)
        mask_rhs = df_encoded[rhs].all(axis=1)
        obs = (mask_lhs & mask_rhs).sum()
        ant = mask_lhs.sum()
        con = mask_rhs.sum()
        exp = (ant * con) / len(df_encoded)
        ic, ic_sd, ic_low, ebgm, eb05 = calc_ic_ebgm(obs, exp)
        enriched.append({**r, "obs": obs, "exp": exp, "IC": ic, "IC_Lower": ic_low, "EBGM": ebgm, "EB05": eb05})

    out_df = pd.DataFrame(enriched)
    out_name = os.path.join(OUTPUT_FOLDER, f"Rules_Target_{col.replace(' ', '_').lower()}.csv")
    out_df.to_csv(out_name, index=False)
    print(f"✅ Saved: {out_name}")

    # plotting
    top_rules = out_df.sort_values(by='lift', ascending=False).head(20)
    if not top_rules.empty:
        # Barplot
        plt.figure(figsize=(12, 6))
        top_rules['rule_str'] = top_rules['lhs'] + " → " + top_rules['rhs']
        sns.barplot(data=top_rules, x="lift", y="rule_str", palette="viridis")
        plt.title(f"Top 20 Rules (Lift) for {col}")
        plt.xlabel("Lift")
        plt.ylabel("Rule")
        plt.tight_layout()
        barplot_path = os.path.join(OUTPUT_FOLDER, f"Rules_Target_{col.replace(' ', '_').lower()}_barplot.png")
        plt.savefig(barplot_path)
        plt.close()

        # Heatmap
        heatmap_data = top_rules.pivot_table(
            index="lhs", columns="rhs", values="lift", aggfunc="max", fill_value=0
        )
        if not heatmap_data.empty:
            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5, linecolor="gray")
            plt.title(f"Heatmap (Top 20 Rules) for {col}")
            plt.xlabel("ADR")
            plt.ylabel("Antecedent")
            plt.tight_layout()
            heatmap_path = os.path.join(OUTPUT_FOLDER, f"Rules_Target_{col.replace(' ', '_').lower()}_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()

    # IC Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(out_df['IC'], bins=20, kde=True, color='skyblue')
    plt.title(f"IC Score Distribution: {col}")
    plt.xlabel("Information Component (IC)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"IC_Histogram_{col.replace(' ', '_').lower()}.png"))
    plt.close()

    # EBGM Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(out_df['EBGM'], bins=20, kde=True, color='coral')
    plt.title(f"EBGM Score Distribution: {col}")
    plt.xlabel("EBGM")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"EBGM_Histogram_{col.replace(' ', '_').lower()}.png"))
    plt.close()
