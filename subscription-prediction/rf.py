import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

file_path = 'shopping_trends.csv' 
shopping_trends_data = pd.read_csv(file_path)

encoded_data = shopping_trends_data.copy()
le = LabelEncoder()

categorical_columns = [
    'Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color',
    'Season', 'Subscription Status', 'Payment Method', 'Shipping Type',
    'Discount Applied', 'Promo Code Used', 'Preferred Payment Method',
    'Frequency of Purchases'
]

for column in tqdm(categorical_columns, desc="Codificação"):
    if column in encoded_data.columns:
        encoded_data[column] = le.fit_transform(encoded_data[column])

# Definir variáveis independentes (X) e alvo (y)
X = encoded_data.drop(columns=['Customer ID', 'Subscription Status'], errors='ignore')
y = encoded_data['Subscription Status']

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

kfold_results = {'No': [], 'Yes': [], 'Overall': []}
kfold_cm_sum = np.zeros((2, 2))  # Inicializar matriz de confusão acumulada

print("Realizando validação cruzada")
with tqdm(total=kfold.get_n_splits(), desc="K-Fold Progress") as pbar:
    for train_idx, val_idx in kfold.split(X, y):
        # Dividir os dados em treino e validação
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Balancear as classes usando SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_fold, y_train_fold)

        # Treinar o modelo
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train_balanced, y_train_balanced)

        # Fazer previsões
        y_val_pred = model.predict(X_val_fold)

        # Atualizar a soma das matrizes de confusão
        kfold_cm_sum += confusion_matrix(y_val_fold, y_val_pred, labels=[0, 1])

        # Calcular métricas
        no_precision = precision_score(y_val_fold, y_val_pred, pos_label=0)
        yes_precision = precision_score(y_val_fold, y_val_pred, pos_label=1)
        no_recall = recall_score(y_val_fold, y_val_pred, pos_label=0)
        yes_recall = recall_score(y_val_fold, y_val_pred, pos_label=1)
        no_f1 = f1_score(y_val_fold, y_val_pred, pos_label=0)
        yes_f1 = f1_score(y_val_fold, y_val_pred, pos_label=1)
        overall_accuracy = accuracy_score(y_val_fold, y_val_pred)

        # Armazenar resultados
        kfold_results['No'].append({'Precision': no_precision, 'Recall': no_recall, 'F1-Score': no_f1})
        kfold_results['Yes'].append({'Precision': yes_precision, 'Recall': yes_recall, 'F1-Score': yes_f1})
        kfold_results['Overall'].append(overall_accuracy)

        # Atualizar barra de progresso
        pbar.update(1)

# Calcular as médias do K-Fold
no_metrics = pd.DataFrame(kfold_results['No']).mean().to_dict()
yes_metrics = pd.DataFrame(kfold_results['Yes']).mean().to_dict()
overall_accuracy = np.mean(kfold_results['Overall'])

# Média das matrizes de confusão
kfold_cm_mean = kfold_cm_sum / kfold.get_n_splits()

# Exibir resultados médios do K-Fold
print("RANDOM FOREST")
print("\nResultados Médios (K-Fold):")
print(f"Classe No:")
print(f"    Precision Média: {no_metrics['Precision']:.4f}")
print(f"    Recall Médio: {no_metrics['Recall']:.4f}")
print(f"    F1-Score Médio: {no_metrics['F1-Score']:.4f}")
print(f"Classe Yes:")
print(f"    Precision Média: {yes_metrics['Precision']:.4f}")
print(f"    Recall Médio: {yes_metrics['Recall']:.4f}")
print(f"    F1-Score Médio: {yes_metrics['F1-Score']:.4f}")
print(f"Acurácia Média: {overall_accuracy:.4f}")

# Plotar a matriz de confusão média do K-Fold
plt.figure(figsize=(8, 6))
sns.heatmap(kfold_cm_mean, annot=True, fmt=".2f", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Matriz de Confusão Média (K-Fold)")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# Divisão para Holdout
X_train_final, X_holdout, y_train_final, y_holdout = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Balancear as classes no Holdout
X_train_balanced, y_train_balanced = SMOTE(random_state=42).fit_resample(X_train_final, y_train_final)

# Treinar o modelo no conjunto de treino completo
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train_balanced, y_train_balanced)

# Fazer previsões no conjunto Holdout
y_holdout_pred = model.predict(X_holdout)

# Calcular métricas para Holdout
holdout_no_precision = precision_score(y_holdout, y_holdout_pred, pos_label=0)
holdout_yes_precision = precision_score(y_holdout, y_holdout_pred, pos_label=1)
holdout_no_recall = recall_score(y_holdout, y_holdout_pred, pos_label=0)
holdout_yes_recall = recall_score(y_holdout, y_holdout_pred, pos_label=1)
holdout_no_f1 = f1_score(y_holdout, y_holdout_pred, pos_label=0)
holdout_yes_f1 = f1_score(y_holdout, y_holdout_pred, pos_label=1)
holdout_overall_accuracy = accuracy_score(y_holdout, y_holdout_pred)

# Exibir resultados para Holdout
print("\nResultados (Holdout):")
print(f"Classe No:")
print(f"    Precision: {holdout_no_precision:.4f}")
print(f"    Recall: {holdout_no_recall:.4f}")
print(f"    F1-Score: {holdout_no_f1:.4f}")
print(f"Classe Yes:")
print(f"    Precision: {holdout_yes_precision:.4f}")
print(f"    Recall: {holdout_yes_recall:.4f}")
print(f"    F1-Score: {holdout_yes_f1:.4f}")
print(f"Acurácia: {holdout_overall_accuracy:.4f}")

# Matriz de Confusão para Holdout
cm_holdout = confusion_matrix(y_holdout, y_holdout_pred, labels=[0, 1])
sns.heatmap(cm_holdout, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Matriz de Confusão - Holdout')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()