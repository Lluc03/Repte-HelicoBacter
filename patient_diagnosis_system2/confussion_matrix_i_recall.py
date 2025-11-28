import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Carrega el CSV generat pel pipeline
pred_df = pd.read_csv("/fhome/maed02/proj_repte3/results/pipeline_ALL_patients/results/ALL_patient_predictions.csv")

y_true = pred_df["true_label"].values       # 0 = negatiu, 1 = positiu
y_pred = pred_df["predicted_label"].values

# 1) Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# 2) Classification report (inclou precision, recall i f1)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["NEGATIU", "POSITIU"]))

# També pots extreure els recalls manualment:
recall_neg = cm[0,0] / (cm[0,0] + cm[0,1])
recall_pos = cm[1,1] / (cm[1,1] + cm[1,0])

print(f"\nRecall NEGATIU (HP-): {recall_neg:.4f}")
print(f"Recall POSITIU (HP+): {recall_pos:.4f}")

# ============================================================
# GUARDAR LA MATRIU DE CONFUSIÓ
# ============================================================

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["NEGATIU", "POSITIU"],
            yticklabels=["NEGATIU", "POSITIU"])
plt.xlabel("Predicció")
plt.ylabel("Real")
plt.title("Confusion Matrix")

output_path = "/fhome/maed02/proj_repte3/results/pipeline_ALL_patients/results/confusion_matrix.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"\nImatge de la matriu de confusió guardada a:\n  {output_path}")
