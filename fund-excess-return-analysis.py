import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

df = pd.read_csv("data/sample_funds.csv")         
df_size = pd.read_csv("data/sample_company_size.csv")  
df88 = pd.read_csv("data/sample_company_data.csv") 
df = df.rename(columns={
    "基金发起人": "sponsor",
    "超额收益区间": "excess_bucket",      
    "基金规模(亿元)": "fund_size_billion"
})
df_size = df_size.rename(columns={
    "基金发起人": "sponsor",
    "最新管理规模(亿)": "aum_billion"
})
print(df.head())

#########################################################################################################

df_fund = df[['sponsor', 'excess_bucket', 'fund_size_billion']]
# Merge fund sponsor with latest AUM (in billions)
df_merged = df_fund.merge(df_size[['sponsor', 'aum_billion']], on='sponsor', how='left')
# Drop rows without company AUM info
df_merged = df_merged[pd.notna(df_merged['aum_billion'])]
# Classify company size by latest AUM (in billions)
def classify_by_latest_size(size):
    if size < 100:
        return 'Small'
    elif size < 1000:
        return 'Medium'
    else:
        return 'Large'
df_merged['company_size'] = df_merged['aum_billion'].apply(classify_by_latest_size)
# ✅ Note: Outperform/Underperform are measured relative to the benchmark return
valid_returns = [
    'Outperform 0–10%',   
    'Underperform 0–10%',  
    'Outperform >10%',     
    'Underperform >10%'    
]
df_merged = df_merged[df_merged['excess_bucket'].isin(valid_returns)]
# Construct combined column (Company size + Excess return bucket)
df_merged['combo'] = df_merged['company_size'] + ' + ' + df_merged['excess_bucket']

# Predefine all 12 combinations (ensure all categories appear, even if count=0)
company_sizes = ['Small', 'Medium', 'Large']
return_ranges = [
    'Outperform 0–10%',
    'Underperform 0–10%',
    'Outperform >10%',
    'Underperform >10%'
]
all_combinations = [f"{size} + {ret}" for size in company_sizes for ret in return_ranges]
# Count frequency of each combination
group_counts = df_merged['combo'].value_counts().reindex(all_combinations, fill_value=0)
# Calculate percentages (rounded to 2 decimals)
percentages = (group_counts / total * 100)
rounded = percentages.round(2)
# Adjust floating-point errors to ensure total = 100%
difference = 100 - rounded.sum()
if abs(difference) >= 0.01:
    max_idx = rounded.idxmax()
    rounded[max_idx] += difference
# Assemble final result DataFrame
result_df = pd.DataFrame({
    'Combination': all_combinations,
    'Fund Count': group_counts.values,
    'Proportion (%)': percentages.values
})

result_df.to_excel("results/company_size_excess_combo_stats.xlsx", index=False)

data = [
    [3637, 6992],  # Large companies: Outperform 0–10% vs. others
    [779, 1788],   # Medium companies
    [78, 322]      # Small companies
]
chi2, p, dof, expected = chi2_contingency(data)

# Output results
print(f"Chi-square: {chi2:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p:.4f}")
# Significance test
alpha = 0.05
if p < alpha:
    print("Result significant: Company size is related to the probability of Outperform 0–10%")
else:
    print("Result not significant: No significant relationship between company size and Outperform 0–10%")
#########################################################################################################

# Calculate terciles (quantiles at 1/3 and 2/3) for company
low = df_size['aum_billion'].quantile(1/3)
high = df_size['aum_billion'].quantile(2/3)
# Print tercile boundaries
print("Tercile cutoffs (AUM in billions):")
print(f"Small company: <= {low:.2f} B")
print(f"Medium company: > {low:.2f} B and <= {high:.2f} B")
print(f"Large company: > {high:.2f} B")

# Classify company size based on terciles
def classify_company(size):
    if size <= low:
        return 'Small'
    elif size <= high:
        return 'Medium'
    else:
        return 'Large'

df_size['company_size'] = df_size['aum_billion'].apply(classify_company)

# Merge company size info back to fund-level data
df_fund = df[['sponsor', 'excess_bucket', 'fund_size_billion']]
df_merged = df_fund.merge(df_size[['sponsor', 'company_size']], on='sponsor', how='left')
df_merged = df_merged.dropna(subset=['company_size'])

valid_returns = [
    'Outperform 0–10%',
    'Underperform 0–10%',
    'Outperform >10%',
    'Underperform >10%'
]
df_merged = df_merged[df_merged['excess_bucket'].isin(valid_returns)]

# Construct combined column
df_merged['combo'] = df_merged['company_size'] + ' + ' + df_merged['excess_bucket']

# Enumerate all combinations
company_sizes = ['Small', 'Medium', 'Large']
return_ranges = [
    'Outperform 0–10%',
    'Underperform 0–10%',
    'Outperform >10%',
    'Underperform >10%'
]
all_combinations = [f"{size} + {ret}" for size in company_sizes for ret in return_ranges]
total = len(df_merged)

# Count occurrences
group_counts = df_merged['combo'].value_counts().reindex(all_combinations, fill_value=0)

# Compute proportions
percentages = (group_counts / total * 100).round(2)
diff = 100 - percentages.sum()
if abs(diff) >= 0.01:
    percentages[percentages.idxmax()] += diff  # adjust for rounding error

# 输出结果
result_df = pd.DataFrame({
    '组合': all_combinations,
    '基金数量': group_counts.values,
   '占比（%）': percentages.values
})
print("\nCompany size × Excess return bucket (benchmark-relative):")
print(result_df)

print("Number of unique sponsors by company size:")
print(df_merged.drop_duplicates(['sponsor'])['company_size']
      .value_counts()
      .reindex(['Small', 'Medium', 'Large']))

# Save results (use relative path in public repo)
result_df.to_excel("results/tercile_companysize_excess_stats.xlsx", index=False)

# Test 2: Company size vs. Outperform 0–10%
data = [
    [3530, 7099],   # Large companies
    [792, 1775],    # Medium companies
    [173, 227]      # Small companies


chi2, p, dof, expected = chi2_contingency(data)

if p < 0.05:
    print("Result significant: Company size is associated with probability of Outperform 0–10%")
else:
    print("Result not significant: No significant association")

# Test 3: Company size vs. Underperform >10%
data = [
    [2567, 8062],   # Large companies
    [786, 1781],    # Medium companies
    [301, 99]       # Small companies
]
chi2, p, dof, expected = chi2_contingency(data)

if p < 0.05:
    print("Result significant: Company size is associated with probability of Underperform >10%")
else:
    print("Result not significant: No significant association")
#########################################3.基金规模分析################################################################
df = df[['fund_size_billion', 'excess_bucket']]
df = df.dropna()
df['fund_size_billion'] = pd.to_numeric(df['fund_size_billion'], errors='coerce')

# Compute terciles (Q1 and Q3)
q1 = df['fund_size_billion'].quantile(1/3)
q3 = df['fund_size_billion'].quantile(2/3)

# Print cutoffs
print("Fund size tercile cutoffs (in billions):")
print(f"Small funds: <= {q1:.2f} B")
print(f"Medium funds: > {q1:.2f} B and <= {q3:.2f} B")
print(f"Large funds: > {q3:.2f} B")

# Classification function
def classify_fund_size(size):
    if size <= q1:
        return 'Small'
    elif size <= q3:
        return 'Medium'
    else:
        return 'Large'

# Add classification column
df['fund_size_category'] = df['fund_size_billion'].apply(classify_fund_size)

valid_returns = [
    'Outperform 0–10%',
    'Underperform 0–10%',
    'Outperform >10%',
    'Underperform >10%'
]
df = df[df['excess_bucket'].isin(valid_returns)]

# Cross-tab: Fund size category × Excess return bucket
cross_tab = pd.crosstab(df['fund_size_category'], df['excess_bucket'], margins=True, margins_name='Total')

# Output counts
print("\n[Counts] Fund size × Excess return bucket:")
print(cross_tab)

# Row percentages
percent_tab = cross_tab.div(cross_tab['Total'], axis=0).round(4) * 100
print("\n[Percentages] Fund size × Excess return bucket (%):")
print(percent_tab)

# --- Fixed Q1/Q3 boundaries (example given explicitly) ---
q1 = 2.12
q3 = 10.91

def classify_fixed(size):
    if size <= q1:
        return 'Small'
    elif size <= q3:
        return 'Medium'
    else:
        return 'Large'

df88['fund_size_billion'] = pd.to_numeric(df88['fund_size_billion'], errors='coerce')
df88 = df88.dropna(subset=['fund_size_billion', 'excess_bucket'])
df88['fund_size_category'] = df88['fund_size_billion'].apply(classify_fixed)

return_types = [
    'Outperform 0–10%',
    'Outperform >10%',
    'Underperform 0–10%',
    'Underperform >10%'
]
df88 = df88[df88['excess_bucket'].isin(return_types)]

# Build crosstab (counts)
count_table = pd.crosstab(df88['fund_size_category'], df88['excess_bucket']).reindex(columns=return_types, fill_value=0)

count_table['总数'] = count_table.sum(axis=1)
percent_table = count_table[return_types].div(count_table['总数'], axis=0).round(4) * 100
percent_table = percent_table.rename(columns=lambda x: f"{x} 比率(%)")
final_df = pd.concat([count_table[return_types], percent_table], axis=1).reset_index()

final_df.to_excel("results/fundsize_excess_distribution.xlsx", index=False)

data = [
    [286, 4246],   # Small funds: Underperform vs Not
    [447, 4087],   # Medium funds
    [412, 4121]    # Large funds
]
chi2, p, dof, expected = chi2_contingency(data)

print("\nChi-square test: Fund size vs. Underperform >10% (benchmark-relative)")
print(f"Chi-square={chi2:.4f}, dof={dof}, p-value={p:.4f}")
if p < 0.05:
    print("Result significant: Fund size is associated with probability of Underperform >10%")
else:
    print("Result not significant: No significant association")

data1 = [
    [1718, 2814],  # Small funds
    [447, 3285],   # Medium funds
    [687, 3846]    # Large funds
]
chi2_1, p_1, dof_1, expected_1 = chi2_contingency(data1)

print("\nChi-square test (alternate demo): Fund size vs. Underperform >10% (benchmark-relative)")
print(f"Chi-square={chi2_1:.4f}, dof={dof_1}, p-value={p_1:.4f}")
if p_1 < 0.05:
    print("Result significant: Fund size is associated with probability of Underperform >10%")
else:
    print("Result not significant: No significant association")

#########################################4.基金经理分析################################################################
# Crosstab: rows = manager, cols = excess buckets (fill missing with 0 and fix column order)
manager_summary = pd.crosstab(df_mgr['manager'], df_mgr['excess_bucket']).reindex(columns=valid_buckets, fill_value=0)
# Total funds managed (row sum)
manager_summary['Total Funds Managed'] = manager_summary.sum(axis=1)

# Percentage columns
valid_buckets = [
    'Outperform 0–10%',
    'Outperform >10%',
    'Underperform 0–10%',
    'Underperform >10%'
]
df_mgr = df88[df88['excess_bucket'].isin(valid_buckets)].copy()
for col in valid_buckets:
    pct_col = f'{col} (%)'
    # Avoid division by zero
    manager_summary[pct_col] = (
        (manager_summary[col] / manager_summary['Total Funds Managed']).where(manager_summary['Total Funds Managed'] > 0, 0) * 100
    ).round(2)

manager_summary = manager_summary.reset_index().rename(columns={'manager': 'Manager'})
manager_summary.to_excel("results/manager_excess_distribution.xlsx", index=False)
#########################################5.基准与输赢################################################################
def split_benchmark(text):
    if pd.isna(text):
        return []
    text = str(text).replace('＋', '+')  # normalize Chinese plus sign to English '+'
    # Keep only items with explicit weight indicator "*"
    return [item.strip() for item in text.split('+') if '*' in item]

df['benchmark_components'] = df['benchmark_string'].apply(split_benchmark)
# Expand rows: one per benchmark component per fund
rows = []
for _, row in df.iterrows():
    components = row['benchmark_components']
    excess_bucket = row['excess_bucket']  # e.g., "Outperform >10%", "Underperform >10%"
    for comp in components:
        rows.append({'Benchmark Component': comp, 'Excess Bucket': excess_bucket})

expanded_df = pd.DataFrame(rows)
summary = (
    expanded_df
    .groupby(['Benchmark Component', 'Excess Bucket'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
summary.to_excel("results/benchmark_component_excess_distribution.xlsx", index=False)