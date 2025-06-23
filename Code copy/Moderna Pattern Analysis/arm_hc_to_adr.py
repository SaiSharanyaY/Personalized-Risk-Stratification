
import pandas as pd
import numpy as np
import os
import ast
import re
from scipy.stats import gamma, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from fpdf import FPDF

# Settings
MIN_SUPPORT = 0.0005
MIN_CONFIDENCE = 0.2
MIN_LIFT = 1
OUTPUT_FOLDER = "Code/Moderna Pattern Analysis/ARM_HC_to_ADR"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Preprocessing functions
def clean_text_item(item):
    item = re.sub(r"[^\w\s\-']", '', item)
    return item.strip().lower()

def extract_hc_items(row):
    try:
        items = ast.literal_eval(row)
        return [f"{clean_text_item(hc)} [health condition]" for hc in items if isinstance(hc, str) and hc.strip()]
    except:
        return []

def build_transactions(df_subset, source_items, target_column):
    transactions = []
    for i, (index, row) in enumerate(df_subset.iterrows()):
        source = source_items[i]
        if not source:
            continue
        targets = [t.strip().lower() for t in str(row[target_column]).split(',')] if pd.notnull(row[target_column]) else []
        if not targets:
            continue
        transactions.append(source + [f"{t} [{target_column.lower()}]" for t in targets])
    return transactions

def filter_rules_by_type(rules, target_column):
    def is_hc_only(items):
        return all('[health condition]' in i for i in items)
    def is_adr_only(items):
        return all(f"[{target_column.lower()}]" in i for i in items)
    return rules[
        rules['antecedents'].apply(is_hc_only) &
        rules['consequents'].apply(is_adr_only) &
        (rules['lift'] >= MIN_LIFT)
    ].copy()

def calculate_disproportionality(rules, df_encoded):
    records = []
    total = len(df_encoded)
    for _, r in rules.iterrows():
        lhs = [r['lhs']]
        rhs = [r['rhs']]
        mask_lhs = df_encoded[lhs].all(axis=1)
        mask_rhs = df_encoded[rhs].all(axis=1)
        obs = (mask_lhs & mask_rhs).sum()
        ant_count = mask_lhs.sum()
        con_count = mask_rhs.sum()
        exp = (ant_count * con_count) / total
        ic = np.log2((obs + 0.5) / (exp + 0.5))
        var = (1/(obs + 0.5) + 1/(exp + 0.5)) / (np.log(2)**2)
        ic_sd = np.sqrt(var)
        ic_lower = ic - 2 * ic_sd
        a = obs + 1.0
        rate = exp + 1.0
        ebgm = a / rate
        eb05 = gamma.ppf(0.05, a, scale=1/rate)
        records.append({
            **r,
            'obs': obs,
            'exp': exp,
            'IC': ic,
            'IC_SD': ic_sd,
            'IC_Lower': ic_lower,
            'EBGM': ebgm,
            'EB05': eb05
        })
    return pd.DataFrame(records)

def run_chi_square_tests(df_rules, df_encoded, top_n=5):
    print("\n🔬 Chi-square tests on top signals:")
    seen = set()
    for _, r in df_rules.iterrows():
        pair = (r['lhs'], r['rhs'])
        if pair in seen:
            continue
        seen.add(pair)
        lhs = [r['lhs']]
        rhs = [r['rhs']]
        mask_lhs = df_encoded[lhs].all(axis=1)
        mask_rhs = df_encoded[rhs].all(axis=1)
        table = [
            [(mask_lhs & mask_rhs).sum(), (mask_lhs & ~mask_rhs).sum()],
            [(~mask_lhs & mask_rhs).sum(), (~mask_lhs & ~mask_rhs).sum()]
        ]
        chi2, p, _, _ = chi2_contingency(table)
        print(f"{lhs[0]} -> {rhs[0]} | χ² = {chi2:.2f}, p = {p:.4f}")
        if len(seen) >= top_n:
            break

#plotting functions
def visualize_top_rules(df_rules, target_column, save_path):
    if df_rules.empty:
        return []
    safe_name = target_column.replace(' ', '_').lower()
    barplot_file = os.path.join(save_path, f"{safe_name}_top10_lift.png")
    ic_plot_file = os.path.join(save_path, f"{safe_name}_ic_distribution.png")
    ebgm_plot_file = os.path.join(save_path, f"{safe_name}_ebgm_distribution.png")

    top_rules = df_rules.sort_values(by='lift', ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='lift', y=top_rules['lhs'] + " -> " + top_rules['rhs'], data=top_rules, color='steelblue')
    plt.title(f"Top 10 Rules by Lift: {target_column}")
    plt.xlabel("Lift")
    plt.ylabel("Rule")
    plt.tight_layout()
    plt.savefig(barplot_file)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df_rules['IC'], bins=20, kde=True, color='skyblue')
    plt.title(f"Distribution of IC Scores: {target_column}")
    plt.xlabel("Information Component (IC)")
    plt.tight_layout()
    plt.savefig(ic_plot_file)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df_rules['EBGM'], bins=20, kde=True, color='coral')
    plt.title(f"Distribution of EBGM Scores: {target_column}")
    plt.xlabel("EBGM")
    plt.tight_layout()
    plt.savefig(ebgm_plot_file)
    plt.close()

    return [barplot_file, ic_plot_file, ebgm_plot_file]

def generate_pdf_report(pdf_path, target_column, image_files, rules_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, f"ARM + Disproportionality Report: {target_column}", ln=True, align='C')
    pdf.set_font("Arial", size=11)
    pdf.ln(5)
    pdf.multi_cell(0, 8, "Top Rules (LHS → RHS):")
    for i, row in rules_df.head(10).iterrows():
        rule_str = f"{row['lhs']} -> {row['rhs']} | lift: {row['lift']:.2f}, IC: {row['IC']:.2f}, EBGM: {row['EBGM']:.2f}"
        pdf.multi_cell(0, 8, rule_str)
    for img_path in image_files:
        pdf.add_page()
        pdf.image(img_path, w=180)
    pdf.output(pdf_path)

def process_target(df, target_column):
    print(f"\n📍 Processing Target: {target_column}")
    df_target = df[df[target_column].notnull()].copy()
    past_hc_items = df_target['ExtractedPastHC'].apply(extract_hc_items).tolist()
    transactions = build_transactions(df_target, past_hc_items, target_column)
    if not transactions:
        print(f"⚠️ Skipped {target_column} - no valid transactions.")
        return

    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    fi = apriori(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
    rules = association_rules(fi, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules_filtered = filter_rules_by_type(rules, target_column)

    if rules_filtered.empty:
        print(f"⚠️ No strong rules for {target_column}")
        return

    def get_key(row):
        lhs = sorted(list(row['antecedents']))
        rhs = sorted(list(row['consequents']))
        return "|".join(sorted(lhs + rhs))

    rules_filtered["lhs"] = rules_filtered["antecedents"].apply(lambda x: list(x)[0])
    rules_filtered["rhs"] = rules_filtered["consequents"].apply(lambda x: list(x)[0])
    rules_filtered["rule_key"] = rules_filtered.apply(get_key, axis=1)
    rules_filtered = rules_filtered.sort_values(by='lift', ascending=False).drop_duplicates("rule_key")

    enriched = calculate_disproportionality(rules_filtered, df_encoded)
    robust = enriched[(enriched["IC_Lower"] > 0) & (enriched["EB05"] > 1)]

    safe_name = target_column.replace(" ", "_").lower()
    robust_path = os.path.join(OUTPUT_FOLDER, f"Robust_Signals_{safe_name}.csv")
    robust.to_csv(robust_path, index=False)
    print(f"✅ Robust signals saved: {robust_path}")
    run_chi_square_tests(robust, df_encoded)
    image_files = visualize_top_rules(robust, target_column, OUTPUT_FOLDER)
    generate_pdf_report(os.path.join(OUTPUT_FOLDER, f"{safe_name}_report.pdf"), target_column, image_files, robust)

def run_full_pipeline(data_path, target_columns):
    df = pd.read_csv(data_path, low_memory=False)
    for col in target_columns:
        process_target(df, col)

if __name__ == "__main__":
    TARGET_COLUMNS = [
        'Gastrointestinal Issues', 'Pain Syndromes', 'Psychological Disorders',
        'Musculoskeletal Disorders', 'Fever', 'Dermatological Conditions',
        'Neurological Disorders', 'Swelling', 'Injection Site Reaction'
    ]
    run_full_pipeline("Data/Final datasets/moderna_with_labels_dose_1+2.csv", TARGET_COLUMNS)
