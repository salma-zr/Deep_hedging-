# Deep Hedging d'un Put sous perte quadratique

Projet académique Python/PyTorch implémentant un algorithme de **Deep Hedging** inspiré de Bühler, Gonon, Teichmann et Wood (2019), appliqué à la couverture d'un put européen avec prime connue.

Le cadre reprend l'idée centrale de l'article : une stratégie de couverture prévisible est paramétrée par un réseau de neurones et entraînée par descente de gradient à travers des trajectoires simulées. Ici, l'objectif principal est spécialisé à la remarque 3.4 de l'article : la prime `p0` est exogène et la stratégie minimise une perte quadratique du P&L terminal.

## Structure

```text
.
├── Buehler et al. - 2019 - Deep hedging (1).pdf
├── README.md
├── requirements.txt
├── notebooks/
│   └── deep_hedging.ipynb
├── report/
│   ├── rapport.tex
│   └── figures/
└── src/
    ├── __init__.py
    ├── model.py
    ├── simulation.py
    ├── training.py
    └── utils.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Le projet n'a aucune dépendance à Colab ou Kaggle.

## Exécution

Depuis la racine du dépôt :

```bash
jupyter notebook notebooks/deep_hedging.ipynb
```

Le notebook est exécutable de bout en bout. Pour une exécution non interactive :

```bash
jupyter nbconvert --to notebook --execute notebooks/deep_hedging.ipynb --output /tmp/deep_hedging_executed.ipynb
```

Les scripts peuvent aussi être importés directement :

```python
from src.simulation import MarketConfig
from src.model import build_hedger, default_model_configs
from src.training import TrainConfig, train_deep_hedger, evaluate_strategy
```

## Méthodologie

- Sous-jacent : dynamique Black--Scholes discrétisée sous probabilité réelle `P`.
- Payoff : put européen `Z = (K - S_T)^+`.
- Prime connue : par défaut le prix Black--Scholes risque-neutre, remplaçable dans `MarketConfig`.
- Stratégie : positions en action `delta_k`, prévisibles, produites par MLP simple, MLP profond ou LSTM.
- P&L vendeur :

```text
P&L_T = -Z + p0 + sum_k delta_k (S_{k+1}-S_k) - C_T(delta)
```

- Objectif principal : minimiser `E_P[(P&L_T)^2]`.
- Extension : coûts de transaction proportionnels et perte CVaR optionnelle.
- Benchmark : delta hedging Black--Scholes calculé sous la mesure risque-neutre.

## Résultats attendus

Les résultats numériques dépendent de la graine et du budget d'entraînement. Sur le scénario de base, les réseaux doivent apprendre une stratégie proche du delta hedging discret en absence de coûts. Le MLP profond tend à réduire l'erreur quadratique par rapport au MLP simple lorsque le budget d'entraînement est suffisant ; le LSTM est pertinent pour tester l'apport d'une mémoire de trajectoire, mais il n'est pas nécessairement meilleur dans un modèle Markovien Black--Scholes où l'état courant contient déjà l'information suffisante.

Les analyses de robustesse du notebook couvrent :

- strikes différents ;
- volatilités différentes ;
- maturités différentes ;
- nombre de dates de couverture ;
- estimation d'intervalles de confiance Monte Carlo ;
- réduction de variance par échantillonnage antithétique.

## Rapport

Le rapport LaTeX `report/rapport.tex` suit la structure imposée :

1. Introduction
2. Cadre mathématique
3. Méthodologie Deep Hedging
4. Implémentation
5. Résultats numériques
6. Analyse de robustesse
7. Extensions
8. Conclusion

Il peut être compilé avec :

```bash
cd report
pdflatex rapport.tex
```

Les figures générées par le notebook sont sauvegardées dans `report/figures/`.
