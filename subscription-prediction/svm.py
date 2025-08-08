import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

file_path = 'shopping_trends.csv' 
shopping_trends_data = pd.read_csv(file_path)

# Codificar variáveis categóricas em numéricas
encoded_data = shopping_trends_data.copy()
le = LabelEncoder()

for column in encoded_data.select_dtypes(include=['object']).columns:
    encoded_data[column] = le.fit_transform(encoded_data[column])

# Definir variáveis independentes (X) e dependente (y)
X = encoded_data.drop(columns=['Customer ID', 'Subscription Status'], errors='ignore')
y = encoded_data['Subscription Status']

# Configurar normalizador e modelo
scaler = StandardScaler()
svm_model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)

# Configurar validação cruzada com K-Fold estratificado
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Variáveis para armazenar resultados do K-Fold
kfold_results = {'No': [], 'Yes': [], 'Overall': []}
kfold_cm_sum = np.zeros((2, 2))  # Inicializar matriz de confusão acumulada

print("Realizando validação cruzada (K-Fold)...")
with tqdm(total=kfold.get_n_splits(), desc="K-Fold Progress") as pbar:
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), start=1):
        # Dividir os dados em treino e validação
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Normalizar os dados
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)

        # Treinar o modelo
        svm_model.fit(X_train_fold_scaled, y_train_fold)

        # Fazer previsões
        y_val_pred = svm_model.predict(X_val_fold_scaled)

        # Atualizar soma das matrizes de confusão
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
print("SUPORT VECTOR MACHINE")
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

# Normalizar os dados para Holdout
X_train_final_scaled = scaler.fit_transform(X_train_final)
X_holdout_scaled = scaler.transform(X_holdout)

# Treinar o modelo no conjunto de treino completo
svm_model.fit(X_train_final_scaled, y_train_final)

# Fazer previsões no conjunto Holdout
y_holdout_pred = svm_model.predict(X_holdout_scaled)

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