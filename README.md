# Fund-Excess-Return-Analysis
Analysis of excess returns in active mutual funds, exploring how “fund size, fund managers, and benchmark composition” relate to performance.


Key dimensions include:
- Fund size: tercile classification (Small / Medium / Large)
- Excess return buckets: Outperform 0–10%, Outperform >10%, Underperform 0–10%, Underperform >10%
- Fund managers: distribution of outcomes by manager
- Benchmark composition: how benchmark weights (e.g., equity, bond, gold) link to excess return performance

🔍 Methods
- Data cleaning and normalization (pandas)
- Cross-tabulation and chi-square tests (scipy)
- Visualization of performance distributions (matplotlib)
- Benchmark component parsing and aggregation

📂 Repo Structure
- `data/` : anonymized or synthetic sample datasets  
- `src/` : Python scripts for classification, cross-tabs, chi-square tests  
- `results/` : output Excel files and figures  

⚠️ Notes
All datasets in this repository are **synthetic or anonymized**.  
Original proprietary data is **not included**.
