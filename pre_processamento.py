import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# 1. Carregando os 3 arquivos csv's
print("Carregando os arquivos...")
df_nondoh = pd.read_csv("Dados/l1-nondoh.csv")
df_benign = pd.read_csv("Dados/l2-benign.csv")
df_malicious = pd.read_csv("Dados/l2-malicious.csv")

# 2. Colando tudo em um arquivo só
print("Atribuindo label às classes...")
df_nondoh['Label'] = 'Non-DoH'
df_benign['Label'] = 'Benign-DoH'
df_malicious['Label'] = 'Malicious-DoH'

print("Juntando os dados...")
df_bruto = pd.concat([df_nondoh, df_benign, df_malicious], ignore_index=True)

# 3. Removendo os metadados de rede
colunas_inuteis = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp']
df_limpo = df_bruto.drop(columns=colunas_inuteis, errors='ignore')

# 4. Limpando as sujeiras matemáticas (NaN e Infinitos)
print("Limpando sujeiras...")
df_limpo = df_limpo.replace([np.inf, -np.inf], np.nan)
df_limpo = df_limpo.dropna()

# 5. Separarando as características (X) do gabarito (y) para não normalizar o gabarito
X = df_limpo.drop(columns=['Label'])
y = df_limpo['Label']

# 6. Aplicando a normalização min-max nas características
print("Normalizando os dados...")
scaler = MinMaxScaler()
X_normalizado = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 7. Colando o gabarito de volta nas características normalizadas
df_final = pd.concat([X_normalizado, y.reset_index(drop=True)], axis=1)

# 8. Renomeando a coluna de gabarito para 'Attack_label'
df_final = df_final.rename(columns={'Label': 'Attack_label'})

# 9. Salvando tudo na pasta "data"
os.makedirs("data", exist_ok=True)
df_final.to_csv("data/new_processed.csv", index=False)

print("Número de características salvas:", len(X.columns))
print("O arquivo 'data/new_processed.csv' foi gerado com sucesso e está pronto pro modelo!")