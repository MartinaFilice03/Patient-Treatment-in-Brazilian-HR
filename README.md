# 🏥 Patient-Treatment-Brazilian-HR  
### Advanced Process Mining Analysis

---

## 📋 Project Overview

This project performs an advanced **Process Mining analysis** on a real-world dataset comprising **3094 clinical records** from a Brazilian Public Hospital (**HR - Hospitais de Rede**). By leveraging the **PM4Py** framework, the analysis transforms raw, noisy hospital logs into actionable strategic insights within the **SUS (Unified Health System)** context.

The goal is to move beyond simple descriptive statistics to identify **structural bottlenecks**, **resource imbalances** ("Hero Culture"), and **non-stationary behaviors** that jeopardize patient safety and operational efficiency.

---

## 🚀 Key Insights & Findings

Based on the analysis of 443 valid end-to-end patient traces (comprising 3,094 clinical events):

* **⚠️ The "Acuity 2" Paradox:** Contrary to clinical intuition, **Urgent patients (Acuity 2)** exhibit the **highest Lead Time** (avg. 0.32 days). While Acuity 1 (Critical with 0.25 days) cases are fast-tracked, Acuity 2 cases represent the system's primary diagnostic bottleneck, likely due to intensive resource competition.
* **📊 Arrival "Burstiness":** The **Dotted Chart** reveals a highly non-linear arrival pattern. These "waves" of events cause temporary system collapses, indicating that staffing must be adjusted to intake peaks rather than daily averages to prevent congestion.
* **📉 Optimal Model Quality:** Through an iterative **K-Tuning optimization**, the model reached an **F1-Score of 0.937** (Fitness: 0.922, Precision: 0.952). This confirms a highly faithful representation of the Brazilian clinical flow, capable of generalizing behaviors without losing accuracy.
* **⏳ The "Long Tail" Problem:** **Violin Plots** show that while standard protocols (Var 1) are stable, less frequent variants suffer from extreme temporal dispersion, often exceeding the **99th percentile threshold (the cases that pass 5.17 hours are considered statistical outliers)**—the so-called "Zombie Cases".
* **🧑‍⚕️ "Hero Culture" & Resource Risk:** Resource analysis (Social Network Analysis) shows a dangerous reliance on specific **"Médico Responsável"** roles. This "Hero Culture" represents a Single Point of Failure (SPOF) where process speed depends on individual performance rather than standardized flow.

---

## 🛠️ The Pipeline

The analysis follows a rigorous **Knowledge Uplift Trail**:

1.  **Ingestion & Mapping:** Converting Brazilian HR data into XES standard attributes (`case:concept:name`, `concept:name`, `time:timestamp`).
2.  **Clinical Cleansing:** Filtering biological outliers and removing "Zombie Cases".
3.  **Process Discovery:** Generating **Process Trees** via **Inductive Miner** to ensure a sound, deadlock-free model.
4.  **Conformance Checking:** Performing intensive **Alignment-based validation** (Log Alignment & Precision computing) with 100% completion across all variants.
5.  **Optimization:** Iteratively testing 10 noise thresholds (K-tuning) to find the perfect balance between Fitness and Precision.

---

## 💻 Getting Started

### Prerequisites

You need **Python 3.x**. Since some systems (e.g., macOS with Homebrew) prevent global package installation, create a virtual environment and install the required libraries inside it:
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas matplotlib seaborn numpy pm4py
```

### Usage

Place your dataset file (e.g., `brazilian_hr_log.csv`) in the root directory and run the analysis script:

```bash
python Patient-Treatment.py
```

⚠️ Note:
This project was developed using Google Colab.  If you want to run it locally, remove the Google Drive mounting  and update the dataset path accordingly.

### 📊 Visualizations

The script generates several critical plots to visualize the process "health":

| Category | Chart Type | Analytical Purpose |
| :--- | :--- | :--- |
| **Process Models** | **Process Tree** | Discovers a sound, hierarchical model using **Inductive Miner** to avoid "spaghetti" flows. |
| **Process Models** | **DFG (Directly Follows)** | Highlights the main clinical pathways and frequent transitions between activities. |
| **Temporal Analysis** | **Histogram + KDE** | Analysis of **Lead Time** distribution and identification of the **99th percentile** threshold. |
| **Temporal Analysis** | **Violin Plots** | Detailed performance comparison of the **Top-10 variants** showing density and stability. |
| **Bottlenecks** | **Acuity Boxplots** | Lead Time stratification by **Acuity Level** (identifying the Acuity 2 bottleneck). |
| **Bottlenecks** | **Dotted Chart** | Visualizes **Arrival Rate** and event density to detect "Burstiness" and intake peaks. |
| **Optimization** | **Line Chart** | **Fitness/Precision/F1 Score** trade-off analysis used to select the optimal **K=7** threshold. |
| **Compliance** | **Progress Bars** | Visual confirmation of 100% completion for Log Alignments and Conformance Checking. |

## 🧪 Methodological Highlights

* **Data-Aware Filtering:** identified and removed **8,289 "dirty" records** (biological inconsistencies and incomplete traces) to ensure discovery algorithms work on high-quality evidence.
* **Lead Time Stabilization:** Removed **"Zombie Cases"** to prevent statistical noise and stabilize the Lead Time distribution.
* **Optimal Modeling (K=7):** Iteratively tested 10 noise thresholds to achieve a **Fitness of 0.922** and a **Precision of 0.952**, maximizing the **F1-Score (0.937)**.
* **Resource Risk:** Detected a **"Hero Culture"** where clinical throughput heavily depends on specific "Médico Responsável" resources, creating a Single Point of Failure.

---

*Analysis performed using Python & PM4Py for the Brazilian Healthcare (HR) Scenario.*
