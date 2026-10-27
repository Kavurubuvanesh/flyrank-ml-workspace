# Automated Content Refresh Prioritization: Benchmarking Tree-Based Classifiers Against Heuristic Baselines on Search Telemetry

**Author:** Buvanesh Kavuru  
**Track:** Machine Learning — Refresh / Content Opportunity Scoring Lane  
**Repository:** [Kavurubuvanesh/flyrank-ml-workspace](https://github.com/Kavurubuvanesh/flyrank-ml-workspace)  
**Capstone Notebooks:** [`work/notebooks/capstone.ipynb`](../work/notebooks/capstone.ipynb) | [`work/notebooks/w08_capstone.ipynb`](../work/notebooks/w08_capstone.ipynb)

---

## 1. Abstract
Can supervised machine learning models significantly outperform deterministic spreadsheet heuristics when prioritizing decaying web pages for editorial refresh across multi-client search portfolios? To answer this question, we evaluate a domain-standard rule-based scorer (`CTR-Fix`) against Logistic Regression, Decision Tree, and Random Forest classifiers using anonymized search telemetry from the FlyRank warehouse (`v20260703` schema cut, 28,703 cleaned content items). All candidate models are trained and benchmarked under a strict 80/20 grouped client-holdout validation split (`GroupShuffleSplit`) to prevent cross-domain authority leakage and client-level baseline memorization. On the held-out client cohort, our Random Forest classifier boosts top-queue review accuracy from a baseline `Precision@50` of `0.240` to `0.740` on the reference benchmark, delivering a ~3.1x improvement in actionable triage precision. These findings demonstrate that non-linear tree ensembles capture rank-dependent click-through rate (CTR) expectations far more effectively than static thresholds, enabling a high-precision, reason-coded decision-support queue for content operations.

---

## 2. Introduction & Problem Statement
Large-scale digital publishers and SEO engineering teams manage hundreds of thousands of indexed URLs, making comprehensive manual audits impossible under weekly bandwidth constraints. **The core operational decision this work supports is weekly editorial queue triage (`Precision@50`)**: determining which 50 specific pages a content engineering team should inspect and refresh first each week to recover decaying organic clicks or capture under-realized search visibility.

Historically, content teams rely on deterministic spreadsheet filters—such as flagging pages with high impressions but low click-through rates (`CTR < 3%`). However, static threshold rules suffer from a fundamental flaw: organic click curves decay exponentially from Position 1 to Position 10. Applying a flat CTR cutoff across all Page-1 ranks systematically misclassifies healthy pages sitting at positions 8–10 while missing subtle decay at positions 2–4. This capstone engineers an end-to-end, leak-free prioritization pipeline (`w04` through `w08`) that replaces brittle cutoffs with calibrated probability ranking and human-in-the-loop operational playbooks.

---

## 3. Data
### 3.1 Warehouse Release & Star Schema
This research builds on the **FlyRank Internship Pseudonymized Warehouse Release (`v20260703`, frozen snapshot export date `2026-07-03`)** hosted on Hugging Face ([FlyRank/internship-warehouse](https://huggingface.co/datasets/FlyRank/internship-warehouse)) alongside the bundled anonymized modeling cut (`data/raw/content_refresh_anonymized.csv`, 28,703 cleaned content items). The upstream warehouse follows a four-table star schema with salted, namespaced cryptographic hash keys:

| Warehouse Table | Row Count | Grain & Role in Pipeline |
| :--- | :--- | :--- |
| `dim_clients` | 104 | One row per pseudonymized client (`client_hash_id`), tracking `is_active`, `has_gsc_access`, `has_ga4_access`, `access_profile`, and telemetry start dates (`gsc_data_start`, `ga4_data_start`). |
| `dim_content` | 519,606 | One row per pseudonymized content item (`content_hash_id`), capturing structural content attributes. |
| `fact_content_daily_performance` | 78,835,655 | Daily grain per `report_date`, `client_hash_id`, and `content_hash_id` partitioned by `month=YYYY-MM`. |
| `fact_content_query_90d` | 2,414,248 | Query-level performance aggregated over a fixed 90-day window with `last_30d` vs. `prev_30d` sub-windows for momentum calculation. |

### 3.2 Date Windows & Exclusions (Public-Safe Hygiene)
* **Observation Windows:** Features aggregate search performance over rolling 30-day (`last_30d`) and prior 30-day (`prev_30d`) sub-windows within the 90-day evaluation frame prior to the `2026-07-03` snapshot.
* **Client & Telemetry Exclusions:** In `dim_clients`, clients with `access_profile IN ('no_search_or_analytics_access', 'source_only_missing_client_dimension', 'ga4_only')` or `has_gsc_access = false` are excluded because reliable Search Console impression and position telemetry is required to evaluate organic search decay.
* **Row-Level Hygiene:** Content items with zero impressions (`impressions_last_30d <= 0`) or invalid rank averages (`avg_position < 1.0`) are removed prior to modeling (yielding 28,703 clean items from the ~30,000 raw starter slice) to prevent zero-division distortion in CTR and delta ratios.
* **Privacy & Anonymization Compliance:** In strict adherence to `DATA_USE.md`, all identifiers remain salted hashes (`client_hash_id`, `content_hash_id`). No raw URLs, client domain names, page titles, private queries, or raw CSV exports are exposed or committed to version control.

---

## 4. Methodology
### 4.1 Exploratory Signal Verification (`w04_baseline_score.ipynb`)
Before constructing predictive models, we audited two domain hypotheses using quartile bucket tables:
1. **Signal 1 — Rank Position vs. CTR (`Verdict: CONFIRMED`):** Grouping pages into average position quartiles confirmed a steep monotonic drop in mean CTR as rank number increases. Consequently, a low CTR at a top rank (`position <= 10`) is a verified statistical anomaly.
2. **Signal 2 — Impression Volume vs. Position (`Verdict: MIXED`):** High impression quartiles still exhibited mean positions above 10, proving that high search demand frequently sits trapped on Page 2 (`positions 11–20`), representing prime "striking-distance" refresh candidates.

### 4.2 Baseline Heuristic Design (`CTR-Fix` Rule)
Our transparent baseline rule (`w04_baseline_score.ipynb` / `02_baseline_score.py`) flags pages satisfying three conjunctive conditions:
* `avg_position <= 10` (Page-1 visibility)
* `impressions_last_30d > 1000` (sufficient traffic volume to justify editorial cost)
* `ctr < 0.03` (underperforming snippet capture)

Flagged candidates (326 pages in the starter cohort) are assigned reason code `LOW_CTR_HIGH_VISIBILITY` and ranked descending by impression volume.

### 4.3 Feature Engineering, Target Definition & Progressive Models (`w05_model.ipynb`)
* **Feature Matrix (`X`):** Incorporates visibility volume (`impressions_last_30d`), rank tier (`avg_position`), engagement efficiency (`ctr`), click volume, and temporal momentum deltas while programmatically stripping all ID hashes, target labels, and group keys.
* **Target Definition (`y`):** Binary indicator identifying verified high-priority refresh opportunities based on rank-adjusted CTR underperformance, striking-distance decay (`positions 4–18`), and traffic recovery potential (`01_prepare_features.py`).
* **Model Progression:** To test whether non-linear complexity is justified, we benchmark three progressive estimators wrapped in leak-free `scikit-learn` `Pipeline` objects (`SimpleImputer(strategy='median')` fit solely on training folds):
  1. **Logistic Regression:** Standardized linear classifier testing additive signal weights.
  2. **Decision Tree (`max_depth=5, min_samples_leaf=25`):** Shallow interpretable tree capturing step-function cutoffs.
  3. **Random Forest (`n_estimators=200, max_depth=8, min_samples_leaf=15`):** Bagged ensemble stabilizing top-50 probability rankings.

### 4.4 Validation Design & Leakage Audit (`w06_validation_audit.ipynb`)
Pages belonging to the same client domain share domain authority, brand recognition, and site template structure. A random row split (`train_test_split`) leaks client-specific baselines across train and test sets.
* **Grouped Holdout Split:** We enforce an 80% train / 20% test **`GroupShuffleSplit` grouped by client cohort**. Every client evaluated in the test split is unseen during training.
* **Leakage Audit:** In `w06_validation_audit.ipynb`, we demonstrated two forms of leakage to guard against: (1) random row splitting (which inflates precision by memorizing client priors), and (2) deterministic formula leakage (where defining a synthetic target directly from unmasked threshold features yields an artificial `1.000` tree ROC-AUC). Both are eliminated in our final grouped benchmark.

---

## 5. Results
All methods were evaluated on the **exact same held-out client split** using **Precision@K** (`K = 25, 50, 100`) and **ROC-AUC**.

### 5.1 Model vs. Baseline Quantitative Comparison

| Scoring Method | Validation Split | `Precision@25` | `Precision@50` | `Precision@100` | Held-Out `ROC-AUC` | Relative Lift (`P@50`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Week-4 Heuristic Baseline (`CTR-Fix`)** | Grouped Client-Holdout | `0.280` | `0.240` | `0.210` | `0.614` | `1.0x (Reference)` |
| **Logistic Regression** | Grouped Client-Holdout | `0.560` | `0.520` | `0.490` | `0.782` | `2.2x` |
| **Decision Tree (`max_depth=5`)** | Grouped Client-Holdout | `0.720` | `0.680` | `0.640` | `0.835` | `2.8x` |
| **Random Forest (`200 trees`)** | Grouped Client-Holdout | **`0.760`** | **`0.740`** | **`0.710`** | **`0.869`** | **`3.1x`** |

### 5.2 Visual Charts: Queue Precision & Feature Importance
*(Full Matplotlib visual figures are rendered inside [`work/notebooks/capstone.ipynb`](../work/notebooks/capstone.ipynb) and [`work/notebooks/w08_capstone.ipynb`](../work/notebooks/w08_capstone.ipynb)).*

#### Chart 1: Held-Out `Precision@50` Comparison (Same Grouped Client Split)

| Model / Scoring Tier | Visual `Precision@50` Bar (0.00 to 1.00 Scale) | `Precision@50` | True Positives in Top 50 |
| :--- | :--- | :---: | :---: |
| **Week-4 Heuristic Rule** | `████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` | **0.240** | 12 / 50 pages |
| **Logistic Regression** | `██████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░` | **0.520** | 26 / 50 pages |
| **Decision Tree (d=5)** | `██████████████████████████████████░░░░░░░░░░░░░░░░` | **0.680** | 34 / 50 pages |
| **Random Forest (200)** | `█████████████████████████████████████░░░░░░░░░░░░░` | **0.740** | **37 / 50 pages** |

#### Chart 2: Random Forest Permutation Importance (Held-Out Test Split, Mean ROC-AUC Drop)

| Telemetry Feature | Relative Importance Bar (Mean ROC-AUC Drop) | Mean Drop ± Std |
| :--- | :--- | :---: |
| `avg_position` | `████████████████████████████████████████████` | `0.214 ± 0.008` |
| `ctr` | `████████████████████████████████████░░░░░░░░` | `0.176 ± 0.006` |
| `impressions_last_30d` | `███████████████████████████░░░░░░░░░░░░░░░░░` | `0.131 ± 0.005` |
| `clicks_last_30d` | `██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` | `0.048 ± 0.003` |
| `position_delta / momentum` | `██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` | `0.029 ± 0.002` |

### 5.3 Error Analysis & Interpretation
* **Why Tree Ensembles Beat the Rule by ~3.1x:** As shown in Chart 2, `avg_position`, `ctr`, and `impressions_last_30d` dominate permutation importance. Whereas the Week-4 rule applies a rigid `ctr < 0.03` cutoff across all Page-1 positions, the Decision Tree and Random Forest partition expected CTR dynamically by rank tier—correctly flagging a 4.5% CTR at Position 2 as severe underperformance while recognizing a 2.5% CTR at Position 9 as normal behavior.
* **False Positive Anatomy:** Inspecting the top-50 queue errors (`13/50` false positives in the reference benchmark) revealed that false alarms cluster on high-impression pages sitting at borderline rank transitions (`avg_position` between `9.2` and `11.5`) where noisy weekly impression spikes temporarily depressed CTR.

---

## 6. Limitations & Honest Framing
In strict compliance with our Week-6 Research Claim Audit (`w06_validation_audit.ipynb`), **all model outputs represent observed, directional signals designed exclusively for human decision-support**:
1. **No Causal or Algorithmic Claims:** This model does not reverse-engineer Google's proprietary ranking algorithm, nor does it prove that refreshing a flagged page will causally recover rankings or traffic. Downstream recovery depends on editorial execution and competitive SERP shifts.
2. **SERP Layout & Zero-Click Blindness:** Search Console telemetry cannot observe on-page SERP features such as AI Overviews, Featured Snippets, People Also Ask boxes, or paid ad density. A page flagged for `LOW_CTR_HIGH_VISIBILITY` may be losing clicks to an AI Overview that answers the query directly on Google rather than suffering from a weak title tag.
3. **Unbalanced Panel & Seasonality:** As documented in `dim_clients`, per-client historical depth varies across the warehouse. Sudden seasonal impression surges can depress short-term CTR without indicating structural content decay.

---

## 7. Ranked Recommendations (Action Playbook)
To operationalize model probabilities for SEO engineering teams, our Week-7 Action Playbook (`w07_action_playbook.ipynb`) maps every flagged page to a deterministic **Reason Code**, **Prescribed Editorial Action**, and **Human-in-the-Loop Guardrail**:

| Priority Tier | Reason Code | Trigger Conditions | Recommended Editorial Playbook |
| :---: | :--- | :--- | :--- |
| **Tier 1** | `LOW_CTR_HIGH_VISIBILITY` | High model score + `avg_position <= 10` + `impressions_last_30d > 1000` + `ctr < 0.03` | **`UPDATE_META_TAGS`**: Core topical authority is already proven by Page-1 placement. Audit the live SERP for zero-click snippets; rewrite `<title>` and meta description to sharpen click value proposition. |
| **Tier 2** | `STRIKING_DISTANCE_DECAY` | High model score + `10 < avg_position <= 20` + `impressions_last_30d > 1000` | **`CONTENT_REFRESH_AND_INTERNAL_LINKS`**: Page sits on the cusp of Page 1 ("quick-win" zone). Update stale sections, expand entity coverage, and add internal anchor links from high-traffic hub pages. |
| **Tier 3** | `SEVERE_POSITION_DROP` | High model score + `avg_position > 20` + `impressions_last_30d > 500` + negative click delta | **`STRUCTURAL_REWRITE_OR_CONSOLIDATE`**: Perform manual search-intent and keyword cannibalization audit. Either execute a comprehensive rewrite or merge into a stronger canonical URL. |

### Operational Guardrails & The "No-Go" Automation List
* **Mandatory Human-in-the-Loop Review:** Every page in the top-50 weekly queue must be inspected on the live Google SERP by an editor before modifying production content.
* **Never Automate `301` Redirects or URL Pruning:** Automated content consolidation risks destroying backlink equity, breaking internal site architecture, and erasing seasonal pages during off-peak months.
* **Never Auto-Publish Unreviewed LLM Metadata:** Generative rewrites of `<title>` tags must pass human brand-voice and factual verification before deployment.
* **Monitoring & Retrain Triggers:** Monitor weekly CTR distributions by rank bucket for macro SERP drift (e.g., Google Core Updates). Retrain the model quarterly or immediately if human reviewer acceptance (`Precision@50`) falls below `0.50`.

---

## 8. Reproducibility
The complete analytical chain—from raw signal audits to playbook generation—is publicly reproducible in the project repository:
* **GitHub Repository:** [Kavurubuvanesh/flyrank-ml-workspace](https://github.com/Kavurubuvanesh/flyrank-ml-workspace)
* **Week 1 — Research Question:** [`work/notebooks/w01_research_question.ipynb`](../work/notebooks/w01_research_question.ipynb)
* **Week 2 — ML Task Framing:** [`work/notebooks/w02_ml_task_framing.ipynb`](../work/notebooks/w02_ml_task_framing.ipynb)
* **Week 3 — Data Contract:** [`work/notebooks/w03_data_contract.ipynb`](../work/notebooks/w03_data_contract.ipynb)
* **Week 4 — Baseline Action Score & Signal Audit:** [`work/notebooks/w04_baseline_score.ipynb`](../work/notebooks/w04_baseline_score.ipynb)
* **Week 5 — Capstone Modeling Benchmark:** [`work/notebooks/w05_model.ipynb`](../work/notebooks/w05_model.ipynb)
* **Week 6 — Validation & Leakage Audit:** [`work/notebooks/w06_validation_audit.ipynb`](../work/notebooks/w06_validation_audit.ipynb)
* **Week 7 — Content Action Playbook:** [`work/notebooks/w07_action_playbook.ipynb`](../work/notebooks/w07_action_playbook.ipynb)
* **Week 8 — Unified Capstone & Charts Notebook:** [`work/notebooks/capstone.ipynb`](../work/notebooks/capstone.ipynb) and [`work/notebooks/w08_capstone.ipynb`](../work/notebooks/w08_capstone.ipynb)
* **Automated Script Pipeline:** Run `python scripts/run_all.py` from the repository root to execute `01_prepare_features.py`, `02_baseline_score.py`, `03_train_model.py`, `04_evaluate_and_export.py`, and `05_build_pdf_report.py`.

---

## 9. Acknowledgments & Data Credit
Built on the FlyRank ML Internship dataset — [FlyRank](https://flyrank.ai). Special thanks to the FlyRank AI engineering and mentorship team for providing the pseudonymized search warehouse ([FlyRank/internship-warehouse](https://huggingface.co/datasets/FlyRank/internship-warehouse)) and reproducible evaluation framework.
