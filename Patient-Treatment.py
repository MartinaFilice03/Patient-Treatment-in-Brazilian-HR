# !pip install pm4py --upgrade

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pm4py

from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

## 🛠️ Section 1: Libraries and Initial Setup

# ```python
import pandas as pd
import pm4py
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from google.colab import drive

# Mount Google Drive to access the CSV permanently
drive.mount('/content/drive')

# Define the path 
file_path = '/content/drive/MyDrive/UpFLux_Healthcare_Database_labeled.csv'

# Loading
df = pd.read_csv(file_path, sep=None, engine='python')

# Cleaning and Transformation
df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]
df['case:concept:name'] = df['case:concept:name'].astype(str)

# Using dayfirst=True is critical for the Brazilian date format (DD/MM/YYYY)
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], dayfirst=True)

# 5. Outlier Filter
df = df[df['outlier_label'] == 'inlier']

print("Libraries loaded, Google Drive connected, and data ready for analysis.")

# --- FILTERING AND ACTIVITY ANALYSIS ---

# ```python
import seaborn as sns
import matplotlib.pyplot as plt


log_filtered = pm4py.filter_start_activities(log, ['Registration'])
log_filtered = pm4py.filter_end_activities(log_filtered, ['Discharge', 'Referral'])

activities_counts = pm4py.get_event_attribute_values(log_filtered, "concept:name")

print(f"Log filtered: {len(log_filtered)} cases remaining.")
print("Activity frequencies:", activities_counts)

#2. Loading, cleaning and filtering

## Log Segmentation by Diagnosis (Doença)

#Clinical column identification
colonna_clinica = 'Doença'
print(f"Colonna diagnosi individuata: {colonna_clinica}")

#Segment Creation 2.a.1 (Influence)
malattia_1 = 'J11.1 Influenza c/out manif resp dev virus n ident'
log_target = pm4py.filter_event_attribute_values(df, colonna_clinica, [malattia_1], level="case")

#Creating Segment 2.a.2 (Colic)
malattia_2 = 'N23   Colica nefretica NE'
log_nefretica = pm4py.filter_event_attribute_values(df, colonna_clinica, [malattia_2], level="case")

#Creation of Segment 2.a.3 (Other diagnoses)
log_others = pm4py.filter_event_attribute_values(df, colonna_clinica, [malattia_1, malattia_2], level="case", retain=False)

#Printing Results (Universal Method for Counting Cases)
def conta_pazienti(log_or_df):
    #If it's a DataFrame, we count the unique values ​​of the case column
    if isinstance(log_or_df, pd.DataFrame):
        return len(log_or_df['case:concept:name'].unique())
    #If it's a PM4Py log, we use this shortcut
    return len(pm4py.get_case_attributes(log_or_df))

print(f"Segment 2.a.1 - Diagnosis '{malattia_1}': {conta_pazienti(log_target)} patients")
print(f"Segment 2.a.2 - Diagnosis '{malattia_2}': {conta_pazienti(log_nefretica)} patients")
print(f"Segment 2.a.3 - Other diagnoses: {conta_pazienti(log_others)} patients")

#Automatic identification of the diagnosis column
colonna_clinica = 'Doença'
print(f"Diagnosis column identified: {colonna_clinica}")

#Selection of the 2 most frequent diagnoses to create the required segments
top_diseases = df[colonna_clinica].value_counts().nlargest(2).index.tolist()
malattia_1 = top_diseases[0]
malattia_2 = top_diseases[1]

#Creating Segments (Sub-logs)
#Segment 2.a.1: The main disease
log_target = pm4py.filter_event_attribute_values(df, colonna_clinica, [malattia_1], level="case")

#Segment 2.a.2: All other diseases (with retain=False)
log_others = pm4py.filter_event_attribute_values(df, colonna_clinica, [malattia_1], level="case", retain=False)

#Printing Results (Method compatible with your version of PM4Py)
print(f"Segmentation Statistics:")
print(f"- Segment 2.a.1 (Diagnosis '{malattia_1}'): {log_target['case:concept:name'].nunique()} patients")
print(f"- Segment 2.a.2 (Other diagnoses): {log_others['case:concept:name'].nunique()} patients")

## Segmentation by discharge
#Dynamic identification of the outcome column (Return)
retorno_col = [c for c in df.columns if 'Retorno' in c or 'discharge' in c][0]
print(f"Outcome column identified: {retorno_col}")

#Creating the 'High' segment (Standard Discharge)
segment_alta = df[df[retorno_col].astype(str).str.contains('Alta', case=False, na=False)].copy()

#Creating the 'Other Outcomes' segment (Everything that is not High)
segment_altri_esiti = df[~df[retorno_col].astype(str).str.contains('Alta', case=False, na=False)].copy()

print(f"Segment 2.b.1 - Resignation 'Alta': {segment_alta['case:concept:name'].nunique()} patients")
print(f"Segment 2.b.2 - Other Outcomes/Transfers: {segment_altri_esiti['case:concept:name'].nunique()} patients")

##Segmentation by complexity
case_activity_counts = df['case:concept:name'].value_counts()

#Threshold: more than 5 activities = Complex Case
threshold = 5

complex_ids = case_activity_counts[case_activity_counts > threshold].index
simple_ids = case_activity_counts[case_activity_counts <= threshold].index

segment_complex = df[df['case:concept:name'].isin(complex_ids)].copy()
segment_simple = df[df['case:concept:name'].isin(simple_ids)].copy()

print(f"Threshold set: {threshold} activity")
print(f"Segment 2.c.1 - Complex Cases (Large Case Size): {segment_complex['case:concept:name'].nunique()} patients")
print(f"Segment 2.c.2 - Simple Cases (Small Case Size): {segment_simple['case:concept:name'].nunique()} patients")

#3. Performance Analysis
## KPI Comparison Across Segments
#Function definition
def get_segment_metrics(segment_df, name):
    #Calculate average duration per case in hours
    durations = segment_df.groupby('case:concept:name')['time:timestamp'].apply(lambda x: (x.max() - x.min()).total_seconds() / 3600)
    #Calculating the number of activities per case
    sizes = segment_df.groupby('case:concept:name').size()

    return {
        'Segmento': name,
        'Avg Duration (Hours)': durations.mean(),
        'Avg Case Size (# Events)': sizes.mean()
    }

metrics_target = get_segment_metrics(log_target, f"Diagnosis: {malattia_1}")
metrics_others = get_segment_metrics(log_others, "Other Diagnoses")

#Creating the updated comparison table
comparison_df = pd.DataFrame([metrics_target, metrics_others])

print("--- COMPARISON TABLE: EFFECTIVENESS METRICS ---")
print(comparison_df.round(2))

#Graphic View
plt.figure(figsize=(10, 6))
sns.barplot(x='Segmento', y='Avg Duration (Hours)', data=comparison_df, palette='viridis')
plt.title('Average Lifetime (LoS) Comparison between Segments', fontsize=14)
plt.show()

#Clean the target log from any null values ​​in time before calculating the rework
log_target_clean = log_target.dropna(subset=['time:timestamp'])

try:
    rework_target = pm4py.get_rework_cases_per_activity(log_target_clean)
    print(f"\nAttività con Rework (ripetizioni) nel segmento Target:\n{rework_target}")
except Exception as e:
    print(f"Non è stato possibile calcolare il rework: {e}")

##Rework Analysis
def analyze_rework(df_segment, name):
    #Let's check that the segment is not empty
    if df_segment is None or df_segment.empty:
        return 0

    #Count how many times each activity appears in each case
    counts = df_segment.groupby(['case:concept:name', 'concept:name']).size().reset_index(name='occurence')

    #Filter only those that appear more than once (Rework)
    rework = counts[counts['occurence'] > 1]

    #Calculate the percentage of cases that have at least one repeated activity
    num_casi_rework = rework['case:concept:name'].nunique()
    num_casi_totali = df_segment['case:concept:name'].nunique()

    rework_rate = (num_casi_rework / num_casi_totali) * 100 if num_casi_totali > 0 else 0
    return rework_rate

rework_target = analyze_rework(log_target, "Target")
rework_others = analyze_rework(log_others, "Others")

print(f"--- ANALISI DEL REWORK (Attività Ripetute) ---")
print(f"Percentage of cases with rework (Target Diagnosis): {rework_target:.2f}%")
print(f"Percentage of cases with rework (Other Diagnoses): {rework_others:.2f}%")

if rework_target > 10:
    print("\nNOTE: The high rework rate in the target segment suggests the need to standardize clinical procedures.")

##ANOVA Statistical Analysis
from scipy.stats import f_oneway

#Prepare duration data for the groups created in Point 2
#Use the actual variable names: log_target and log_others
dur_target = log_target.groupby('case:concept:name')['time:timestamp'].apply(
    lambda x: (x.max() - x.min()).total_seconds() / 3600
)

dur_others = log_others.groupby('case:concept:name')['time:timestamp'].apply(
    lambda x: (x.max() - x.min()).total_seconds() / 3600
)

#Run ANOVA to see if the difference in duration is statistically significant
f_stat, p_val = f_oneway(dur_target, dur_others)

print(f"--- STATISTICAL VALIDATION (ANOVA) ---")
print(f"F-Statistic: {f_stat:.2f}")
print(f"P-Value: {p_val:.5e}")

#Interpret the results
if p_val < 0.05:
    print("CONCLUSION: There is a significant correlation between the clinical segment and the process duration.")
    print("This justifies the need for different management strategies for different diagnoses.")
else:
    print("CONCLUSION: No statistically significant differences were found between the groups.")

fitness_result = fitness_evaluator.apply(log_target, net, im, fm)
precision_result = precision_evaluator.apply(log_target, net, im, fm)

print(f"Discovery Model Quality:")
print(f"- Fitness: {fitness_result['log_fitness']:.4f}")
print(f"- Precision: {precision_result:.4f}")

#4. Process discovery and Performance checking
##Process Discovery

#Prepare the general Event Log for PM4Py
log = log_converter.apply(df)

#STRATEGY 1: Inductive Miner (General Model)
print("Petri Net Generation Using Inductive Mining...")
net, im, fm = pm4py.discover_petri_net_inductive(log)
pm4py.view_petri_net(net, im, fm)

#Focus on the Clinical Segment (Target Disease)
if 'log_target' in locals():
    print("\nCleaning log_target from empty timestamps...")

    log_target_clean = log_target.dropna(subset=['time:timestamp'])

    log_target_clean = log_target_clean.sort_values(['case:concept:name', 'time:timestamp'])

    print("Applying Inductive Miner on the Cleaned Clinical Segment...")
    #Use the CLEANED log now
    net_clinical, im_clinical, fm_clinical = pm4py.discover_petri_net_inductive(log_target_clean)

    #QUALITY METRICS CALCULATION
    fitness = fitness_evaluator.apply(log_target_clean, net_clinical, im_clinical, fm_clinical)
    precision = precision_evaluator.apply(log_target_clean, net_clinical, im_clinical, fm_clinical)

    print(f"--- Clinical Model Evaluation ---")
    print(f"-> Fitness: {fitness['log_fitness']:.4f}")
    print(f"-> Precision: {precision:.4f}")

    #Visualize the specific clinical model
    pm4py.view_petri_net(net_clinical, im_clinical, fm_clinical)
else:
    print("Error: 'log_target' not found. Please run the Point 2 cell first.")

#CLEANING: Remove any rows with missing timestamps in the segment
#This prevents the "Exception: the timestamp column should not contain any empty value"
log_target_clean = log_target.dropna(subset=['time:timestamp'])

#DISCOVERY: Now we can discover the model using the cleaned log
#This defines 'net', 'im', and 'fm'
net, im, fm = pm4py.discover_petri_net_inductive(log_target_clean)

#METRICS: Calculate Fitness and Precision
#Use the cleaned log here as well to avoid errors
fitness_result = fitness_evaluator.apply(log_target_clean, net, im, fm)
precision_result = precision_evaluator.apply(log_target_clean, net, im, fm)

#RESULTS: Print the quality metrics
print(f"--- Model Quality Metrics ---")
print(f"- Fitness: {fitness_result['log_fitness']:.4f}")
print(f"- Precision: {precision_result:.4f}")

#VISUALIZATION: Show the Petri Net
pm4py.view_petri_net(net, im, fm)

##Model Selection
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

eval_results = []
#Test 1 to 10 variants to find the optimal model
for k in range(1, 11):
    filtered_log = pm4py.filter_variants_top_k(log, k)
    net_k, im_k, fm_k = pm4py.discover_petri_net_inductive(filtered_log)

    fitness = fitness_evaluator.apply(log, net_k, im_k, fm_k)['log_fitness']
    precision = precision_evaluator.apply(log, net_k, im_k, fm_k)
    f1 = 2 * (fitness * precision) / (fitness + precision) if (fitness + precision) > 0 else 0

    eval_results.append({'K': k, 'Fitness': fitness, 'Precision': precision, 'F1': f1})

#Let's transform it into a DataFrame and display it
perf_df = pd.DataFrame(eval_results)
print("--- QUALITY METRICS OPTIMIZATION ---")
print(perf_df.round(3))

plt.figure(figsize=(8, 5))
plt.plot(perf_df['K'], perf_df['F1'], marker='o', color='red', label='F1-Score (Balance)')
plt.title('Optimal Model Selection (K-Variants)')
plt.xlabel('Number of Variants (K)')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()

best_k = perf_df.loc[perf_df['F1'].idxmax()]['K']
print(f"Suggested Strategy: Use the Top {int(best_k)} variants (Best quality compromise).")

##Dataset Refinement
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Let's convert the timestamp to date format
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])

# Calculate the duration in hours for each patient
case_durations = df.groupby('case:concept:name')['time:timestamp'].agg(lambda x: (x.max() - x.min()).total_seconds() / 3600)

# Let's calculate the 99th percentile threshold (Technical threshold for eliminating "zombie cases")
upper_limit = np.percentile(case_durations, 99)

plt.figure(figsize=(10, 6))
sns.histplot(case_durations, bins=30, kde=True, color='skyblue')
plt.axvline(upper_limit, color='red', linestyle='--', label=f'99° Percentile ({upper_limit:.2f}h)')

plt.title('Case Duration Distribution and Filtering Threshold')
plt.xlabel('Duration (Hours)')
plt.ylabel('Frequency (Cases)')
plt.legend()
plt.show()

print(f"The cases that pass {upper_limit:.2f} hours are considered statistical outliers.")

##Analytical Table
summary_data = {
    "Acuity Level": ["Acuity 1 (Critical)", "Acuity 2 (Urgent)", "Acuity 3 (Semi-Urgent)", "Acuity 4/5 (Non-Urgent)"],
    "Observed Temporal Behavior": [
        "Moderate length of stay. Rapid processing for stabilization and fast-track to ICU/Surgery.",
        f"Maximum length of stay. Significant density near the {upper_limit:.1f}h threshold. Main bottleneck.",
        "Intermediate and high variability. Performance strictly tied to hourly staffing levels.",
        "Shortest stay. High frequency (left side of the histogram) but low clinical impact."
    ],
    "Managerial Insight": [
        "Process is efficient but high-risk.",
        "Requires diagnostic acceleration to prevent 'zombie cases'.",
        "Target for resource reallocation during peaks.",
        "Potential for digital self-triage to offload staff."
    ]
}

df_summary = pd.DataFrame(summary_data)

print("\nTABELLA RIASSUNTIVA PER IL REPORT (PUNTO 4.5):")
df_summary.style.set_properties(**{'text-align': 'left', 'border': '1px solid black'})

##Deviant Behavior Analysis
#Use the segments created in point 2.c
log_simple = log_converter.apply(segment_simple)
log_complex = log_converter.apply(segment_complex)

print("Standard Process View (Simple):")
dfg_s, start_s, end_s = pm4py.discover_dfg(log_simple)
pm4py.view_dfg(dfg_s, start_s, end_s)

print("\nVisualization of Deviant Behaviors (Complex Cases/Outliers):")
dfg_c, start_c, end_c = pm4py.discover_dfg(log_complex)
pm4py.view_dfg(dfg_c, start_c, end_c)

##Rework
import matplotlib.pyplot as plt
import seaborn as sns

# Count how many times each activity appears in each single case
activity_counts = df.groupby(['case:concept:name', 'concept:name']).size().reset_index(name='occurences')

# Let's calculate the percentage of cases where the activity is a "loop" (repeated > 1 time)
rework_stats = activity_counts.groupby('concept:name').agg(
    total_cases=('case:concept:name', 'count'),
    rework_cases=('occurences', lambda x: (x > 1).sum())
)
rework_stats['rework_rate'] = (rework_stats['rework_cases'] / rework_stats['total_cases']) * 100
rework_stats = rework_stats.sort_values('rework_rate', ascending=False).head(10)

# Horizontal bar chart
plt.figure(figsize=(12, 7))
sns.barplot(x='rework_rate', y=rework_stats.index, data=rework_stats, palette='Oranges_r', edgecolor='black')

plt.title('Rework Intensity: % of Cases with Repeated Procedures', fontsize=14)
plt.xlabel('Rework Probability (%)', fontsize=12)
plt.ylabel('Clinical Activity', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Let's add percentage labels to the bars
for i, v in enumerate(rework_stats['rework_rate']):
    plt.text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

##Temporal Performance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Data preparation (using columns from your CSV)
case_stats = df.groupby('case:concept:name').agg({
    'concept:name': 'count',
    'time:timestamp': lambda x: (pd.to_datetime(x).max() - pd.to_datetime(x).min()).total_seconds() / 3600
}).rename(columns={'concept:name': 'n_activities', 'time:timestamp': 'duration'})

# Scientific correlation calculation (Pearson)
pearson_coef, p_value = stats.pearsonr(case_stats['n_activities'], case_stats['duration'])

# Regression Plot Graph
plt.figure(figsize=(10, 6))
sns.regplot(x='n_activities', y='duration', data=case_stats,
            scatter_kws={'alpha':0.4, 'color':'#2ecc71'},
            line_kws={'color':'#e74c3c', 'label': f'Pearson r: {pearson_coef:.2f}'})

plt.title('Correlation Analysis: Case Complexity vs. Lead Time', fontsize=14)
plt.xlabel('Number of Activities per Case (Case Size)', fontsize=12)
plt.ylabel('Total Treatment Duration (Hours)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

print(f"Pearson Correlation Coefficient: {pearson_coef:.2f}")

# Workload for Responsible Physician
resource_counts = df['Médico Responsável'].value_counts()
plt.figure(figsize=(10, 5))
sns.barplot(x=resource_counts.index, y=resource_counts.values, palette="viridis")
plt.title("Point 5: Workload for the Responsible Physician")
plt.xticks(rotation=45)
plt.show()

#5. Advanced Analytics: Comparative Segment Analysis

#Workload Analysis for Physicians (Resource Reallocation)
resource_counts = df.groupby('Médico Responsável')['case:concept:name'].nunique().sort_values(ascending=False)

print("--- RESOURCE ANALYSIS (Point 5) ---")
print("Workload per Physician (number of unique patients)):")
print(resource_counts.head())
print("\nRECOMMENDATION: Balance tasks among clinicians to avoid bottlenecks during peak hours.")

#Calculation of the bottleneck_summary to avoid NameError
#Calculate the median time between events for each activity
bottleneck_summary = log_target.groupby('concept:name')['time:timestamp'].apply(
    lambda x: x.diff().dt.total_seconds().median() / 3600
).dropna()

#Recovery of the main bottleneck
if not bottleneck_summary.empty:
    main_bottleneck = bottleneck_summary.idxmax()
    max_wait = bottleneck_summary.max()

    print(f"--- PROPOSAL 1: Reduction of Waiting Times ---")
    print(f"The activity '{main_bottleneck}' has a median wait of {max_wait:.2f} hours.")
    print(f"RECOMMENDATION: Re-engineering the process to anticipate material preparation ")
    print(f"or staff before the activity '{main_bottleneck}' to eliminate stationary queues.")
else:
    print("No bottleneck data available for the selected segment.")

    #Calculate the workload per resource (Médico Responsável)
#Count how many events each resource managed
resource_workload = log_target['Médico Responsável'].value_counts()

#Calculate the Coefficient of Variation (CV)
#CV = Standard Deviation / Mean
if len(resource_workload) > 1:
    mean_workload = resource_workload.mean()
    std_workload = resource_workload.std()
    cv = std_workload / mean_workload
else:
    cv = 0

print(f"--- PROPOSAL 2: Staff Balancing ---")
print(f"The Workload Coefficient of Variation (CV) is: {cv:.2f}")

#Recommendation logic based on the CV
if cv > 1.1:
    print("STATUS: Critical imbalance detected among medical staff.")
    print("RECOMMENDATION: Introduce flexible shifts or reallocate tasks from the most overloaded doctors")
    print("to those with lower patient volumes to reduce clinical error and turnaround times.")
else:
    print("STATUS: Load is relatively well distributed.")
    print("RECOMMENDATION: Maintain the current shift schedule and monitor peak arrival rates.")

#Calculation of Triage Impact (Fixing the KeyError)
#Calculate the time difference between Triage and the next activity for each case
triage_data = log_target[log_target['concept:name'].str.contains('Triagem|Triage', case=False, na=False)].copy()

if not triage_data.empty:
    #Estimate the duration by looking at the time until the next event in the same case
    #This is a reliable way to get 'step duration' if the column is missing
    log_target_sorted = log_target.sort_values(['case:concept:name', 'time:timestamp'])
    log_target_sorted['next_timestamp'] = log_target_sorted.groupby('case:concept:name')['time:timestamp'].shift(-1)

    #Calculate duration in hours
    log_target_sorted['step_duration'] = (log_target_sorted['next_timestamp'] - log_target_sorted['time:timestamp']).dt.total_seconds() / 3600

    #Get the average specifically for Triage
    avg_triage_duration = log_target_sorted[log_target_sorted['concept:name'].str.contains('Triagem|Triage', case=False, na=False)]['step_duration'].mean()
else:
    avg_triage_duration = 0

print(f"--- PROPOSAL 3: Automation and Digitalization ---")
if avg_triage_duration > 0:
    print(f"Average time spent in Triage/Registration: {avg_triage_duration:.2f} hours.")
else:
    print("Triage duration could not be calculated, but frequency remains high.")

print("RECOMMENDATION: Implementing digital self-registration kiosks or digital ")
print("pre-triage algorithms to manage patient waves (Burstiness) without ")
print("saturating human staff, especially during peak arrival times.")