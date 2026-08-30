import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
import os

# Set plotting style
plt.style.use('ggplot')
sns.set_palette("husl")
output_dir = "reports/figures"

print("Loading data...")
df = pd.read_csv('data/processed/engineered_running_times.csv')

# --- PLOT 1: Correlation Heatmap ---
print("Generating Correlation Heatmap...")
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.savefig(f"{output_dir}/01_correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# --- PLOT 2: Travel Time by Hour ---
print("Generating Travel Time by Hour Boxplot...")
plt.figure(figsize=(12, 6))
sns.boxplot(x='hour', y='run_time_in_seconds', data=df, palette='viridis')
plt.title('Distribution of Travel Time by Hour of Day')
plt.xlabel('Hour of Day (0-23)')
plt.ylabel('Travel Time (seconds)')
plt.ylim(0, 1000)
plt.savefig(f"{output_dir}/02_travel_time_by_hour.png", dpi=300, bbox_inches='tight')
plt.close()

# --- Train XGBoost for Output Plots ---
print("Training XGBoost Model...")
X = df.drop(columns=['run_time_in_seconds'])
y = df['run_time_in_seconds']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# --- PLOT 3: Actual vs Predicted ---
print("Generating Actual vs Predicted Plot...")
plt.figure(figsize=(8, 8))
plt.scatter(y_test, xgb_preds, alpha=0.2, color='darkorange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs Predicted ETA (XGBoost)')
plt.xlabel('Actual Travel Time (s)')
plt.ylabel('Predicted Travel Time (s)')
plt.xlim(0, 1500)
plt.ylim(0, 1500)
plt.savefig(f"{output_dir}/03_actual_vs_predicted.png", dpi=300, bbox_inches='tight')
plt.close()

# --- PLOT 4: Residuals ---
print("Generating Residuals Distribution Plot...")
residuals = y_test - xgb_preds
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=100, kde=True, color='red')
plt.axvline(x=0, color='black', linestyle='--')
plt.title('Distribution of Prediction Errors (Residuals)')
plt.xlabel('Error in Seconds (Actual - Predicted)')
plt.ylabel('Frequency')
plt.xlim(-500, 500)
plt.savefig(f"{output_dir}/04_residuals_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# --- PLOT 5: SHAP Summary Plot ---
print("Generating SHAP Summary Plot...")
X_test_sample = X_test.sample(2000, random_state=42)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_sample)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.title('SHAP Summary Plot: Feature Impact on ETA')
plt.savefig(f"{output_dir}/05_shap_summary.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"All done! High-resolution graphs saved to {output_dir}/")
