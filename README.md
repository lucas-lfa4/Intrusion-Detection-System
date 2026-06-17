# Detecção de intrusão em tráfego DoH (DNS over HTTPS)

Projeto de desenvolvimento e avaliação de um Sistema de Detecção de Intrusão (IDS) especializado em identificar ataques e tunelamento malicioso ocultos no tráfego DoH (DNS over HTTPS). O projeto está dividido em duas fases principais: a **replicação metodológica** do artigo de referência e a **validação de generalização** utilizando um novo conjunto de dados.

---

## Resumo do artigo de referência

O trabalho replica a metodologia proposta por **Zebin et al. (2022)**. O artigo aborda o desafio de segurança introduzido pelo protocolo DoH que, ao criptografar as requisições DNS dentro do tráfego HTTPS padrão, protege a privacidade do usuário, mas impede que ferramentas tradicionais de segurança detectem atividades maliciosas.

Para resolver esse problema sem descriptografar os pacotes, os autores propõem uma arquitetura de Aprendizado Empilhado (**Stacked Ensemble**). O sistema utiliza três sub-modelos de *Random Forest* na camada base e uma *Regressão Logística* como Meta-Classificador para dar o veredito final (Non-DoH, Benign-DoH ou Malicious-DoH). Para mitigar o desbalanceamento massivo dos dados, o método aplica a técnica SMOTE combinada com a divisão da classe majoritária em blocos paralelos. O artigo original reporta uma acurácia excepcional de **99,9%** e integra valores SHAP para garantir a explicabilidade (XAI) do modelo.

> **Citação Original:**
> *Zebin, T., Rezvy, S., & Al-Anis, A. (2022). A Cyber-Security Intrusion Detection System for DNS over HTTPS (DoH) Attacks Using a Balanced Stacked Ensemble and SHAP Explainer. In Deep Learning Applications, Volume 3. Springer.*

---

## Conjuntos de Dados (datasets)

Este repositório engloba o ciclo completo de testes do sistema, alimentado por duas bases de dados distintas:

### 1. Dataset Original - fase de replicação
* **Nome:** `CIRA-CIC-DoHBrw-2020` (Canadian Institute for Cybersecurity).
* **Composição:** Criado a partir da fusão estratégica dos arquivos brutos `l1-nondoh.csv` (Tráfego HTTPS padrão), `l2-benign.csv` (DoH legítimo) e `l2-malicious.csv` (Ataques e tunelamento). 
* **Características:** Focado em 29 atributos estatísticos e comportamentais do fluxo de rede (como duração do fluxo e tamanho dos pacotes).

### 2. Novo Dataset - fase de generalização e melhoria
* **Nome:** `CIRA-CIC-DoHBrw-2020-and-DoH-Tunnel-Traffic-HKD`, disponível em https://github.com/doh-traffic-dataset/CIRA-CIC-DoHBrw-2020-and-DoH-Tunnel-Traffic-HKD/
* **Objetivo:** É formado pela junção do `CIRA-CIC-DoHBrw-2020`, artigo originalmente utilizado no artigo trabalhado na disciplina, com o `DoH-Tunnel-Traffic-HKD`, que acrescenta instâncias de ataques geradas por 3 novas ferramentas de tunneling: tuns, dnstt, e tcp_over_dns.
* **Estrutura:** O dataset é estruturado em 3 camadas. A primeira classifica o total de instâncias entre NonDoH ou DoH; a segunda é como um "afunilamento" das instâncias DoH da primeira camada, classificando-as entre BenignDoH ou MaliciousDoH; A terceira camda é mais um "afunilamento", mas dessa vez das instâncias MaliciousDoH da segunda camada, classificando-as de acordo com a ferramenta de tunneling que a gerou (dns2tcp, iodine, dnscat2, dnstt, tcp_over_dns, tuns).

---

## Conteúdo do repositório

O repositório está estruturado para apresentar os dois escopos do projeto:

1. **Replicação fiel:** Implementação exata do pipeline do artigo (Divisão em blocos, SMOTE no treino e 10-Fold Cross-Validation Out-of-Fold). Os resultados obtidos em laboratório alcançaram **98% de acurácia global**, validando o alto poder preditivo da arquitetura e identificando os limites reais de recall causados pelo desbalanceamento severo da base original.
2. **Teste com dataset novo e melhorias:** Scripts e análises focados na aplicação do modelo original no novo conjunto de dados. Esta etapa inclui a proposta de otimização computacional por meio da **Redução de Dimensionalidade**, utilizando a importância de atributos extraída pelo SHAP na Fase 1 para remover variáveis irrelevantes.

---

## Equipe: 

- Daniel Nascimento da Silva
- Lucas Ferreira Alves  
- Maria Carolina Santos Berrafato
- Millena Ferreira Marçal das Neves
