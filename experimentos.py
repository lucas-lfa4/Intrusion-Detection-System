"""
# Imports e Datasets

Os dataset utilizados no presente notebook são o DoH-Tunnel-Traffic-HKD dataset, disponível em https://github.com/doh-traffic-dataset/DoH-Tunnel-Traffic-HKD/, e o CIRA-CIC-DoHBrw-2020-and-DoH-Tunnel-Traffic-HKD, disponível em https://github.com/doh-traffic-dataset/CIRA-CIC-DoHBrw-2020-and-DoH-Tunnel-Traffic-HKD/

O segundo dataset é formado pela junção do CIRA-CIC-DoHBrw-2020, artigo originalmente utilizado no artigo trabalhado, com o DoH-Tunnel-Traffic-HKD, que acrescenta instâncias de ataques geradas por 3 novas ferramentas de tunneling.
"""

pip install imbalanced-learn

from google.colab import drive

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

drive.mount('/content/drive')

# Acessando os arquivos doS datasets no Google Drive

caminho_base = '/content/drive/MyDrive/Datasets/Detecção de Intrusão'

df_hkd = pd.read_csv(caminho_base + '/DoH-Tunnel-Traffic-HKD/Total-48h-Augmentation.csv')
df_cira_l1 = pd.read_csv(caminho_base + '/CIRA-CIC-DoHBrw-2020-and-DoH-Tunnel-Traffic-HKD/l1-total-add.csv')
df_cira_l2 = pd.read_csv(caminho_base + '/CIRA-CIC-DoHBrw-2020-and-DoH-Tunnel-Traffic-HKD/l2-total-add.csv')
df_cira_l3 = pd.read_csv(caminho_base + '/CIRA-CIC-DoHBrw-2020-and-DoH-Tunnel-Traffic-HKD/l3-total-add.csv')

"""## Dataset 1: DoH-Tunnel-Traffic-HKD"""

print(f'Shape HKD: {df_hkd.shape}\n')
df_hkd.head()

df_hkd.info()

df_hkd.describe()

# Checando se existem valores faltantes

faltantes = df_hkd.isnull().sum()
faltantes[faltantes > 0].sort_values(ascending=False)

# Confirmando a quantidade de instâncias por classe

df_hkd['Label'].value_counts()

# Visualizando a distribuição de classes

df_hkd['Label'].value_counts().plot(kind='bar')
plt.show()

"""## Dataset 2: CIRA-CIC-DoHBrw-2020-and-DoH-Tunnel-Traffic-HKD

### Layer 1
"""

print(f'Shape CIRA-CIC (layer 1): {df_cira_l1.shape}\n')
df_cira_l1.head()

df_cira_l1.info()

df_cira_l1.describe()

# Checando se existem valores faltantes

missing = df_cira_l1.isnull().sum()
missing[missing > 0].sort_values(ascending=False)

# Confirmando a quantidade de instâncias por classe

df_cira_l1['Label'].value_counts()

# Visualizando a distribuição de classes

df_cira_l1['Label'].value_counts().plot(kind='bar')
plt.show()

"""### Layer 2"""

print(f'Shape CIRA-CIC (layer 2): {df_cira_l2.shape}\n')
df_cira_l2.head()

df_cira_l2.info()

df_cira_l2.describe()

# Checando se existem valores faltantes

faltantes = df_cira_l2.isnull().sum()
faltantes[faltantes > 0].sort_values(ascending=False)

# Confirmando a quantidade de instâncias por classe

df_cira_l2['Label'].value_counts()

# Visualizando a distribuição de classes

df_cira_l2['Label'].value_counts().plot(kind='bar')
plt.show()

"""### Layer 3"""

print(f'Shape CIRA-CIC (layer 3): {df_cira_l3.shape}\n')
df_cira_l3.head()

df_cira_l3.info()

df_cira_l3.describe()

# Checando se existem valores faltantes

faltantes = df_cira_l3.isnull().sum()
faltantes[faltantes > 0].sort_values(ascending=False)

# Confirmando a quantidade de instâncias por classe

df_cira_l3['Label'].value_counts()

# Visualizando a distribuição de classes

df_cira_l3['Label'].value_counts().plot(kind='bar')
plt.show()

"""## Pré-Processamento

### Limpeza

O HKD não possui valores faltantes, como pudemos observar na análise exploratória dos dados. Entretanto, sua junção com o CIRA-CIC possui valores faltantes nas 3 camadas. Como os autores não mencionam como foi realizado o tratamento desses valores, optamos por remover essas instâncias, visto que a quantidade é pouco significativa.
"""

# Deletando as instâncias que possuem valores faltantes

df_cira_l1 = df_cira_l1.dropna()
df_cira_l2 = df_cira_l2.dropna()
df_cira_l3 = df_cira_l3.dropna()

"""No trabalho original, os autores juntam os layers 1 e 2 no dataset CIRA-CIC, unificando as labels. Isso significa que as instâncias DoH no layer 1 (que classifica ataques como DoH ou non-DoH) possuem a classificação "detalhada" no layer 2: as instâncias DoH são anotadas como Normal DoH ou Suspicious DoH. Para juntar os dois datasets, vamos concatenar a segunda camada do dataset apenas às instâncias non-DoH do primeiro, de forma a evitar duplicatas.

Adicionalmente, faremos um teste adicional, já que estamos usando uma versão ainda mais detalhada do CIRA-CIC que possui uma terceira camada: uniremos as instâncias non-DoH da camada 1 às instâncias Normal DoH da camada 2 e à camada 3 completa, que corresponde aos ataques maliciosos de fato.
"""

# Removendo as colunas irrelevantes para a análise, conforme metodologia do artigo (que fica com 29 features no final)

colunas_irrelevantes = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp']

df_hkd_limpo = df_hkd.drop(columns=colunas_irrelevantes)
df_cira_l1_limpo = df_cira_l1.drop(columns=colunas_irrelevantes)
df_cira_l2_limpo = df_cira_l2.drop(columns=colunas_irrelevantes)
df_cira_l3_limpo = df_cira_l3.drop(columns=colunas_irrelevantes)

non_doh = df_cira_l1_limpo[df_cira_l1_limpo['Label'] == 'NonDoH']
normal_doh = df_cira_l2_limpo[df_cira_l2_limpo['Label'] == 'Benign']

df_cira_l1_l2 = pd.concat([non_doh, df_cira_l2_limpo])
df_cira_l1_l2_l3 = pd.concat([non_doh, normal_doh, df_cira_l3_limpo])

df_cira_l1_l2['Label'].value_counts()

df_cira_l1_l2_l3['Label'].value_counts()

# Renomeando as labels para maior clareza

df_cira_l1_l2['Label'] = df_cira_l1_l2['Label'].replace({
    'Malicious': 'MaliciousDoH',
    'Benign': 'BenignDoH'
})

df_cira_l1_l2_l3['Label'] = df_cira_l1_l2_l3['Label'].replace({
    'Benign': 'BenignDoH',
    'dns2tcp': 'MaliciousDoH_dns2tcp',
    'iodine': 'MaliciousDoH_iodine',
    'dnstt': 'MaliciousDoH_dnstt',
    'dnscat2': 'MaliciousDoH_dnscat2',
    'tcp-over-dns': 'MaliciousDoH_tcp_over_dns',
    'tuns': 'MaliciousDoH_tuns'
})

"""### Balanceamento e Divisão dos Dados"""

# Proporções das classes para os datasets

contagem_hkd = df_hkd_limpo['Label'].value_counts()
proporcao = (contagem_hkd / contagem_hkd.min()).round().astype(int)
print('HKD Dataset -> ' + ' : '.join(map(str, proporcao.values)))

contagem_cira_l1_l2 = df_cira_l1_l2['Label'].value_counts()
proporcao = (contagem_cira_l1_l2 / contagem_cira_l1_l2.min()).round().astype(int)
print('CIRA L1 + L2  -> ' + ' : '.join(map(str, proporcao.values)))

contagem_cira_l1_l2_l3 = df_cira_l1_l2_l3['Label'].value_counts()
proporcao = (contagem_cira_l1_l2_l3 / contagem_cira_l1_l2_l3.min()).round().astype(int)
print('CIRA L1 + L2 + L3 -> ' + ' : '.join(map(str, proporcao.values)))

X_hkd = df_hkd_limpo.drop(columns=['Label'])
y_hkd = df_hkd_limpo['Label']

X_train_hkd, X_test_hkd, y_train_hkd, y_test_hkd = train_test_split(
    X_hkd,
    y_hkd,
    test_size=0.1,
    stratify=y_hkd,
    random_state=42
)

"""Não é necessário balancear o HKD dataset, mas o CIRA-CIC sim, conforme as proporções identificadas de cada label. Os autores do artigo original dividem o dataset em conjuntos de treino (90%) e teste (10%), e em seguida, dividem o conjunto de treino em porções como estratégia para balanceamento dos dados. Repetindo o processo nas próximas células, teremos:

1. Divisão dos dados em treino e teste

CIRA L1 + L2:

2. Divisão do conjunto de treino em 3 partições de NonDoH
3. Aplicar o SMOTE para aumentar a quantidade de instâncias da classe BenignDoH

CIRA L1 + L2 + L3:

2. Divisão do conjunto de treino em 5 partições de NonDoH
3. Aplicar o SMOTE para aumentar a quantidade de instâncias da classe BenignDoH
"""

# Divisão dos dados em treino (90%) e teste (10%)

X_cira_l1_l2 = df_cira_l1_l2.drop(columns=['Label'])
y_cira_l1_l2 = df_cira_l1_l2['Label']

X_cira_l1_l2_l3 = df_cira_l1_l2_l3.drop(columns=['Label'])
y_cira_l1_l2_l3 = df_cira_l1_l2_l3['Label']

# CIRA L1 + L2
X_train_cira_l1_l2, X_test_cira_l1_l2, y_train_cira_l1_l2, y_test_cira_l1_l2 = train_test_split(
    X_cira_l1_l2,
    y_cira_l1_l2,
    test_size=0.10,
    stratify=y_cira_l1_l2,
    random_state=42
)

# CIRA L1 + L2 + L3
X_train_cira_l1_l2_l3, X_test_cira_l1_l2_l3, y_train_cira_l1_l2_l3, y_test_cira_l1_l2_l3 = train_test_split(
    X_cira_l1_l2_l3,
    y_cira_l1_l2_l3,
    test_size=0.10,
    stratify=y_cira_l1_l2_l3,
    random_state=42
)

# Juntando X e y dos dados em uma cópia para facilitar a divisão dos conjuntos

df_train_cira_l1_l2 = X_train_cira_l1_l2.copy()
df_train_cira_l1_l2['Label'] = y_train_cira_l1_l2

df_train_cira_l1_l2_l3 = X_train_cira_l1_l2_l3.copy()
df_train_cira_l1_l2_l3['Label'] = y_train_cira_l1_l2_l3

# Confirmando a classe com maior quantidade de instâncias (CIRA L1 + L2)

df_train_cira_l1_l2['Label'].value_counts()

# Confirmando a classe com maior quantidade de instâncias (CIRA L1 + L2 + L3)

df_train_cira_l1_l2_l3['Label'].value_counts()

# Particionando os dados em 3 conjuntos: CIRA L1 + L2

classe_majoritaria = df_train_cira_l1_l2[
    df_train_cira_l1_l2['Label'] == 'NonDoH'
].sample(frac=1, random_state=42)

particoes_non_doh_l1_l2 = np.array_split(
    classe_majoritaria,
    3
)

classes_minoritarias = df_train_cira_l1_l2[
    df_train_cira_l1_l2['Label'] != 'NonDoH'
]

subconjuntos_l1_l2 = []

for particao_non_doh in particoes_non_doh_l1_l2:

    subconjunto = pd.concat([particao_non_doh, classes_minoritarias], ignore_index=True)
    subconjuntos_l1_l2.append(subconjunto)

print(f'Quantidade de subconjuntos para o CIRA L1 + L2: {len(subconjuntos_l1_l2)}')

print('Quantidade de instâncias para cada uma das partições (CIRA L1 + L2):\n')

for i, subconjunto in enumerate(subconjuntos_l1_l2):
    print(f"\nSubconjunto {i+1}:")
    print(subconjunto['Label'].value_counts())

# Aplicando o SMOTE no CIRA L1 + L2

subconjuntos_balanceados_cira_l1_l2 = []

for subconjunto in subconjuntos_l1_l2:
    X_sub = subconjunto.drop(columns=['Label'])
    y_sub = subconjunto['Label']

    smote = SMOTE(
        sampling_strategy={
          'BenignDoH': 120000
      }, random_state=42)

    X_bal, y_bal = smote.fit_resample(X_sub, y_sub)

    subconjuntos_balanceados_cira_l1_l2.append((X_bal, y_bal))

# Checando a nova distribuição:

for i, (_, y_bal) in enumerate(subconjuntos_balanceados_cira_l1_l2):

    print(f'\nSubconjunto Balanceado: {i+1}')

    print(pd.Series(y_bal).value_counts())

    contagem_subconjunto = pd.Series(y_bal).value_counts()
    proporcao = (contagem_subconjunto / contagem_subconjunto.min()).round().astype(int)

print('\nNovas proporções CIRA L1 + L2  -> ' + ' : '.join(map(str, proporcao.values)))

# Particionando os dados em 5 conjuntos: CIRA L1 + L2 + L3

classe_majoritaria = df_train_cira_l1_l2_l3[
    df_train_cira_l1_l2_l3['Label'] == 'NonDoH'
].sample(frac=1, random_state=42)

particoes_non_doh_l1_l2_l3 = np.array_split(
    classe_majoritaria,
    5
)

classes_minoritarias = df_train_cira_l1_l2_l3[
    df_train_cira_l1_l2_l3['Label'] != 'NonDoH'
]

subconjuntos_l1_l2_l3 = []

for particao_non_doh in particoes_non_doh_l1_l2_l3:

    subconjunto = pd.concat([particao_non_doh, classes_minoritarias], ignore_index=True)
    subconjuntos_l1_l2_l3.append(subconjunto)

print(f'Quantidade de subconjuntos para o CIRA L1 + L2 + L3: {len(subconjuntos_l1_l2_l3)}')

print('Quantidade de instâncias para cada uma das partições (CIRA L1 + L2 + L3):\n')

for i, subconjunto in enumerate(subconjuntos_l1_l2_l3):
    print(f"\nSubconjunto {i+1}:")
    print(subconjunto['Label'].value_counts())

# Aplicando o SMOTE no CIRA L1 + L2 + L3

subconjuntos_balanceados_cira_l1_l2_l3 = []

for subconjunto in subconjuntos_l1_l2_l3:
    X_sub = subconjunto.drop(columns=['Label'])
    y_sub = subconjunto['Label']

    smote = SMOTE(
        sampling_strategy={
            'MaliciousDoH_iodine': 100000,
            'MaliciousDoH_dnstt': 100000,
            'MaliciousDoH_dnscat2': 100000,
            'MaliciousDoH_tcp_over_dns': 100000,
            'MaliciousDoH_tuns': 100000,
            'BenignDoH': 100000
      }, random_state=42)

    X_bal, y_bal = smote.fit_resample(X_sub, y_sub)

    subconjuntos_balanceados_cira_l1_l2_l3.append((X_bal, y_bal))

# Checando a nova distribuição:

for i, (_, y_bal) in enumerate(subconjuntos_balanceados_cira_l1_l2_l3):

    print(f'\nSubconjunto Balanceado: {i+1}')

    print(pd.Series(y_bal).value_counts())

    contagem_subconjunto = pd.Series(y_bal).value_counts()
    proporcao = (contagem_subconjunto / contagem_subconjunto.min()).round().astype(int)

print('\nNovas proporções CIRA L1 + L2 + L3 -> ' + ' : '.join(map(str, proporcao.values)))

"""Após a estratégia de rebalanceamento e aplicação do SMOTE, chegamos ao seguinte cenário na alteração das proporções:

- CIRA L1 + L2 -> 45 : 18 : 1
- CIRA L1 + L2 + L3 -> 45 : 8 : 2 : 2 : 2 : 2 : 1 : 1

Novas proporções:
- CIRA L1 + L2 -> 3 : 2 : 1 (3 partições)
- CIRA L1 + L2 + L3 -> 2 : 2 : 1 : 1 : 1 : 1 : 1 : 1 (5 partições)

### Normalização
"""

scaler = MinMaxScaler()

X_train_hkd_scaled = scaler.fit_transform(X_train_hkd)
X_test_hkd_scaled = scaler.transform(X_test_hkd)

# CIRA L1 + L2
subconjuntos_normalizados_cira_l1_l2 = []

for X_bal, y_bal in subconjuntos_balanceados_cira_l1_l2:

    scaler = MinMaxScaler()

    X_train_cira_l1_l2_scaled = scaler.fit_transform(X_bal)
    X_test_cira_l1_l2_scaled = scaler.transform(X_test_cira_l1_l2)

    subconjuntos_normalizados_cira_l1_l2.append((X_train_cira_l1_l2_scaled, y_bal, X_test_cira_l1_l2_scaled, scaler))

# CIRA L1 + L2 + L3
subconjuntos_normalizados_cira_l1_l2_l3 = []

for X_bal, y_bal in subconjuntos_balanceados_cira_l1_l2_l3:

    scaler = MinMaxScaler()

    X_train_cira_l1_l2_l3_scaled = scaler.fit_transform(X_bal)
    X_test_cira_l1_l2_l3_scaled = scaler.transform(X_test_cira_l1_l2_l3)

    subconjuntos_normalizados_cira_l1_l2_l3.append((X_train_cira_l1_l2_l3_scaled, y_bal, X_test_cira_l1_l2_l3_scaled, scaler))

"""## Modelos"""

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

resultados_cv_cira_l1_l2 = []
rfs_cira_l1_l2 = []

for X_train_cira_l1_l2_scaled, y_train_bal, X_test_cira_l1_l2_scaled, scaler in subconjuntos_normalizados_cira_l1_l2:

    rf = RandomForestClassifier(
        n_estimators=10,
        max_features=28,
        random_state=42,
        n_jobs=-1
    )

    scores = cross_validate(
        rf,
        X_train_cira_l1_l2_scaled,
        y_train_bal,
        cv=cv,
        scoring=[
            'roc_auc',
            'accuracy',
            'precision',
            'f1',
            'recall'
        ],
        n_jobs=-1
    )

    resultados_cv_cira_l1_l2.append(scores)

    rf.fit(
        X_train_cira_l1_l2_scaled,
        y_train_bal
    )

    rfs_cira_l1_l2.append(rf)

# Gerando meta-features para o modelo

def gerar_meta_features(rfs, subconjuntos_normalizados):
    meta_train_list = []
    meta_test_list  = []
    y_train_all     = []

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for rf, (X_train_scaled, y_train_bal, X_test_scaled, _) in zip(rfs, subconjuntos_normalizados):
        # Out-of-fold predictions no treino
        oof_proba = cross_val_predict(
            rf, X_train_scaled, y_train_bal,
            cv=cv5, method='predict_proba', n_jobs=-1
        )
        meta_train_list.append(oof_proba)
        y_train_all.append(y_train_bal)

        # Predições no conjunto de teste
        meta_test_list.append(rf.predict_proba(X_test_scaled))

    X_meta_train = np.vstack(meta_train_list)
    y_meta_train = np.concatenate(y_train_all)
    # Média das probabilidades dos sub-modelos no teste
    X_meta_test  = np.mean(np.stack(meta_test_list, axis=0), axis=0)

    return X_meta_train, y_meta_train, X_meta_test


print("Gerando meta-features para CIRA L1 + L2 ...")
X_meta_train_l1_l2, y_meta_train_l1_l2, X_meta_test_l1_l2 = gerar_meta_features(
    rfs_cira_l1_l2, subconjuntos_normalizados_cira_l1_l2
)

# CIRA L1 + L2
meta_clf_l1_l2 = LogisticRegression(
    max_iter=1000, multi_class='multinomial', solver='lbfgs',
    random_state=42, n_jobs=-1
)
meta_clf_l1_l2.fit(X_meta_train_l1_l2, y_meta_train_l1_l2)
print("Meta-classificador (CIRA L1 + L2) treinado")

