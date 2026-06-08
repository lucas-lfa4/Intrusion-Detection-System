from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


pkl_dir = Path.cwd() / "pkls"
data_dir= Path.cwd() / "data"
df=pd.read_csv(data_dir /'new_processed.csv')
y=df['Attack_label']
X=df.drop(columns=['Attack_label'])



# classifier
X_train, X_test,y_train, y_test=train_test_split(X,y, stratify=y,test_size = 0.1, random_state=42)

# Técnica do 10-Fold Cross-Validation
print("\nIniciando 10-Fold Cross-Validation... (Isso pode demorar um pouco!)")

# 1. Matriz para o treinamento da regressão logística (3 classes x 3 sub-modelos)
X_meta_train = np.zeros((len(X_train), 9)) 

# 2. Configura o gerador de K-Folds (10 pedaços balanceados)

# Função para embaralhar e dividir a base de dados em 10
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 3. O Loop do 10 Fold Cross-Validation (Treina em 90%, prevê em 10%)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"--- Processando Fold {fold+1}/10 ---")
    
    # Separa os dados de treino e teste DESTA iteração usando as posições (.iloc)
    X_treino_fold, y_treino_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    
    # Divisão do banco de dados em 3:

    # Concatena as 29 metricas com a label
    df_treino_fold = pd.concat([X_treino_fold, y_treino_fold], axis=1)
    
    # Divide em 3 df Non-DoH, Benign e Malicius
    df_nondoh = df_treino_fold[df_treino_fold['Attack_label'] == 'Non-DoH']
    df_benign = df_treino_fold[df_treino_fold['Attack_label'] == 'Benign-DoH']
    df_malicious = df_treino_fold[df_treino_fold['Attack_label'] == 'Malicious-DoH']
    
    # Embaralha o df só da classe majoritária
    df_nondoh = shuffle(df_nondoh, random_state=42)
    tamanho = len(df_nondoh) // 3
    # Lista com 3 df só de Non-DoH
    splits_nondoh = [
        df_nondoh.iloc[:tamanho], 
        df_nondoh.iloc[tamanho:2*tamanho], 
        df_nondoh.iloc[2*tamanho:]
    ]
    
    # Aplicando a técnica SOMOTE para a classe Benign-DoH ficar com o mesmo tamanho de Malicious-DoH
    smote = SMOTE(sampling_strategy={'Benign-DoH': len(df_malicious)}, random_state=42)
    modelos_do_fold = []
    
    # Treinando os 3 sub-modelos nos 90% (X_treino_fold)
    for i in range(3):
        df_bloco = pd.concat([splits_nondoh[i], df_benign, df_malicious])
        X_bloco = df_bloco.drop(columns=['Attack_label'])
        y_bloco = df_bloco['Attack_label']
        
        X_res, y_res = smote.fit_resample(X_bloco, y_bloco)
        
        rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        rf.fit(X_res, y_res)
        modelos_do_fold.append(rf)
    
    # Os três modelos preveem as probabilidades do X_val_fold (dados que eles nunca viram)
    p1 = modelos_do_fold[0].predict_proba(X_val_fold)
    p2 = modelos_do_fold[1].predict_proba(X_val_fold)
    p3 = modelos_do_fold[2].predict_proba(X_val_fold)
    
    # Salva essas previsões na matriz principal exatamente nas posições de validação
    X_meta_train[val_idx] = np.hstack([p1, p2, p3])


# Meta-Classificador:
print("\nTreinando o Meta-Classificador (Chefe)...")
meta_clf = LogisticRegression(random_state=42, max_iter=1000)
meta_clf.fit(X_meta_train, y_train)


# Treinando novamente três sub-modelos oficiais (O trabalho feito anteriormente foi só para treinar o meta-classificador)
print("\nTreinando os Sub-Modelos oficiais com 100% dos dados de treino...")

df_treino_final = pd.concat([X_train, y_train], axis=1)
df_nondoh_final = df_treino_final[df_treino_final['Attack_label'] == 'Non-DoH']
df_benign_final = df_treino_final[df_treino_final['Attack_label'] == 'Benign-DoH']
df_malicious_final = df_treino_final[df_treino_final['Attack_label'] == 'Malicious-DoH']

df_nondoh_final = shuffle(df_nondoh_final, random_state=42)
tam_final = len(df_nondoh_final) // 3
splits_final = [
    df_nondoh_final.iloc[:tam_final], 
    df_nondoh_final.iloc[tam_final:2*tam_final], 
    df_nondoh_final.iloc[2*tam_final:]
]

smote_final = SMOTE(sampling_strategy={'Benign-DoH': len(df_malicious_final)}, random_state=42)
sub_modelos_finais = []

for i in range(3):
    df_bloco_final = pd.concat([splits_final[i], df_benign_final, df_malicious_final])
    X_bloco_final = df_bloco_final.drop(columns=['Attack_label'])
    y_bloco_final = df_bloco_final['Attack_label']
    
    X_res_f, y_res_f = smote_final.fit_resample(X_bloco_final, y_bloco_final)
    
    rf_final = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf_final.fit(X_res_f, y_res_f)
    sub_modelos_finais.append(rf_final)


print("\nPredição final com o uso do Meta-Classificador")
proba1_test = sub_modelos_finais[0].predict_proba(X_test)
proba2_test = sub_modelos_finais[1].predict_proba(X_test)
proba3_test = sub_modelos_finais[2].predict_proba(X_test)
X_meta_test = np.hstack([proba1_test, proba2_test, proba3_test])

y_pred = meta_clf.predict(X_meta_test)

print("\nRelatório de Métricas:")
print(classification_report(y_test, y_pred))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Dashboard
print("\nGerando o Dashboard (Baseado nas features do Sub-modelo 1)...")

class_explainer = ClassifierExplainer(
    sub_modelos_finais[0], X_test, y_test,
    labels=['Non-DoH', 'Benign-DoH', 'Malicious-DoH'] 
)

pkl_dir.mkdir(exist_ok=True)
class_explainer.dump(pkl_dir / "explainer.pkl")

db = ExplainerDashboard(class_explainer)
db.run()

'''
# classifier
model = RandomForestClassifier(n_estimators=10, random_state=42,max_depth=5,class_weight='balanced')
model.fit(X_train, y_train)
class_explainer = ClassifierExplainer(model, X_test, y_test, 
                               labels=['Non-DOH', 'Benign-DoH', 'Malicious-DOH'])
_ = ExplainerDashboard(class_explainer)
#class_explainer.dump(pkl_dir/ "class_explainer.joblib")
class_explainer.dump(pkl_dir/ "explainer.pkl")
'''