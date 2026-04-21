# 🏠 House Prices — Advanced Regression Techniques

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Prédire le prix de vente de maisons à Ames, Iowa à partir de 79 variables.  
Métrique : **RMSLE** (Root Mean Squared Log Error sur SalePrice).

---

## Structure

```
house-prices/
├── data/
│   ├── raw/          # Données Kaggle brutes (non commitées)
│   └── processed/    # Données nettoyées
├── notebooks/        # Jupyter — EDA, features, modèles
├── src/              # Code Python réutilisable
├── submissions/      # Fichiers CSV pour Kaggle
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/TON_USERNAME/house-prices.git
cd house-prices
pip install -r requirements.txt

# Télécharger les données
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/raw/
```

## Étapes

- [ ] 01 — EDA & visualisation
- [ ] 02 — Nettoyage & feature engineering
- [ ] 03 — Modèle baseline
- [ ] 04 — Modèles avancés + ensemble
- [ ] 05 — Soumission Kaggle
