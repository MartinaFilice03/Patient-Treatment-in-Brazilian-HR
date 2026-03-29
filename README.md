Process Mining: Patient Treatment Flow in Brazilian Healthcare (HR)
📋 Project Overview
This repository presents a specialized Process Mining analysis of clinical workflows within a Brazilian Public Hospital (HR - Hospitais de Rede). The dataset contains approximately 25,000 events, representing the journey of patients from Triage to Discharge/Transfer.

The primary objective is to evaluate operational resilience, identify bottlenecks in the Brazilian Unified Health System (SUS) context, and optimize the process model through rigorous Conformance Checking.

🚀 Strategic Insights (Brazilian HR Case)
Based on the analysis of 1,801 end-to-end patient traces:

⚠️ The "Acuity 2" Paradox: In the Brazilian HR context, Urgent patients (Acuity 2) experience the highest Lead Time (avg. 0.32 days). While Acuity 1 (Critical) cases are fast-tracked, Acuity 2 cases suffer from diagnostic queueing, becoming the system's primary bottleneck.

📊 Event Density & "Burstiness": The Dotted Chart highlights a highly non-linear arrival pattern. These "bursts" of patients coincide with shift changes and peak hours, causing temporary system collapses that are visible as vertical clusters in the temporal distribution.

📉 Optimal Model Quality: To represent the Brazilian clinical flow accurately, the model was tuned to K=7, achieving a Fitness of 0.922 and Precision of 0.952. An F1-Score of 0.937 confirms the model is robust enough for administrative decision-making.

⏳ Performance Variance: Violin Plots of the Top 10 variants show that while standard protocols (Var 1) are efficient, non-standard deviations lead to extreme "Long Tail" delays, often exceeding the 99th percentile threshold (~31.4 hours).

🛠️ The Analytical Pipeline
Data Ingestion: Mapping Brazilian clinical attributes to the XES standard (case:concept:name, concept:name, time:timestamp).

Clinical Cleansing: Filtering "Zombie Cases" (extreme outliers) and biological inconsistencies to stabilize the event log.

Process Discovery: Using the Inductive Miner to extract a hierarchical Process Tree that avoids the "spaghetti model" typical of complex hospital logs.

Optimization: Iterative testing of noise thresholds to balance model complexity with empirical reality.

📊 Key Visualizations
Category	Chart	Purpose
Optimization	K-Optimization Table	Summarizing 10 iterations of Fitness/Precision tuning.
Temporal	Lead Time Histogram	Visualizing the 99th percentile cutoff for "Zombie Case" removal.
Variability	Violin Plots (Top 10)	Comparing temporal stability across the most frequent Brazilian clinical paths.
Dynamics	Dotted Chart	Detecting arrival waves and event density over the observation period.
🧪 Methodological Validation
The final selected model (K=7) demonstrates high reliability for the Brazilian HR scenario:

Fitness: 0.922 (Explains 92% of actual patient movements).

Precision: 0.952 (Avoids over-generalization and "ghost" paths).

F1-Score: 0.937 (Strong harmonic balance).