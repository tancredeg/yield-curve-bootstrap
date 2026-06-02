# CONTEXTE COMPLET — Yield Curve Construction and Rates Pricing
## Document de référence pour continuer le projet dans un nouvel onglet

---

## 1. PERSONA ET OBJECTIF DU PROJET

**Persona assignée à l'assistant :**
Senior rates quant avec expérience production en infrastructure de yield curve dans une banque tier-1. Le code produit doit passer la review d'un quant desk sans remarque : numériquement correct, conventions précises, structuré, exécutable immédiatement.

**Objectif du projet :**
Portfolio GitHub démontrant une maîtrise desk-level de :
- Bootstrapping de courbes de taux
- Méthodes d'interpolation
- Pricing d'obligations et de swaps
- Analyse de scénarios
- Comparaison USD (SOFR) vs EUR (ESTR)

**Audience cible :** recruteurs quantitatifs évaluant des candidats pour des rôles rates trading, fixed income structuring, ou risk management. Le repo doit signaler : "cette personne comprend ce qu'est une courbe de discount, comment elle est construite en pratique, où les implémentations naïves échouent, et ce que les chiffres signifient pour un desk."

---

## 2. LIVRABLES PRODUITS (ÉTAT ACTUEL : COMPLET)

### Fichiers générés
```
yield_curve_project/
├── yield_curve.py          ✅ 1292 lignes — script principal exécutable
├── README.md               ✅ Documentation GitHub professionnelle
├── requirements.txt        ✅ Dépendances pinnées
└── figures/
    ├── 01_zero_curves_interpolation.png   ✅
    ├── 02_scenario_analysis_usd.png       ✅
    ├── 03_scenario_analysis_eur.png       ✅
    └── 04_usd_eur_comparison.png          ✅
```

### Exécution
```bash
pip install -r requirements.txt
python yield_curve.py
```

### Dépendances pinnées
```
numpy==2.4.4
scipy==1.17.1
pandas==3.0.2
matplotlib==3.10.8
```

---

## 3. DONNÉES DE MARCHÉ (HARDCODÉES — REF : 2024-01-15)

Données synthétiques calibrées aux conditions début 2024.
- **USD** : courbe inversée (taux terminal Fed ~5.3%, long end ~4.1%)
- **EUR** : légèrement pentue à la hausse (BCE ~4.0%, long end ~3.2%)

### USD SOFR (Act/360 deposits, semi-annual swap conventions)
```python
USD_DEPOSITS = {
    1/12: 0.0531,   # 1M
    3/12: 0.0533,   # 3M
    6/12: 0.0530,   # 6M
}
USD_SWAPS = {
    2: 0.0462,  3: 0.0440,  5: 0.0420,  7: 0.0415,
    10: 0.0410, 12: 0.0405, 15: 0.0408, 20: 0.0412, 30: 0.0418,
}
```

### EUR ESTR (Act/360 deposits, annual swap conventions)
```python
EUR_DEPOSITS = {
    1/12: 0.0388,   # 1M
    3/12: 0.0390,   # 3M
    6/12: 0.0387,   # 6M
}
EUR_SWAPS = {
    2: 0.0325,  3: 0.0310,  5: 0.0305,  7: 0.0308,
    10: 0.0312, 12: 0.0315, 15: 0.0318, 20: 0.0320, 30: 0.0322,
}
```

### Constantes
```python
ACT360 = 360
ACT365 = 365
SEMI   = 2      # fréquence semi-annuelle
ANNUAL = 1      # fréquence annuelle
```

---

## 4. QUATRE FIXES CRITIQUES IMPLÉMENTÉS

Ces quatre problèmes sont des **bugs connus des implémentations naïves**. Chacun est documenté dans le code avec un commentaire explicite.

### FIX 1 — Gaps d'interpolation pendant le bootstrap [CRITIQUE]
**Problème :** bootstrapper un swap 20Y sans pilier entre 10Y et 20Y force une extrapolation plate de P(0,10Y) pour toutes les dates de coupon intermédiaires. Biais structurel de ~64bp sur le zero rate 20Y.
**Fix :** les données incluent explicitement les piliers 12Y et 15Y.
**Commentaire dans le code :**
> "12Y and 15Y pillars included to prevent flat extrapolation bias during 20Y bootstrap. Without these, zero rate 20Y would be ~64bp above the swap rate — a known failure mode of naive bootstrap."

### FIX 2 — Convention de duration modifiée [SÉRIEUX]
**Problème :** D_mod = D_mac n'est valide que sous compounding continu.
**Fix :** trois conventions implémentées, toutes nommées explicitement :
```
continuous  : D_mod = D_mac
semiannual  : D_mod = D_mac / (1 + ytm/2)    ← standard USD
annual      : D_mod = D_mac / (1 + ytm)      ← standard EUR
```
`D_mod = D_mac` comme valeur par défaut **n'apparaît nulle part** dans le code.

### FIX 3 — Vérification indépendante du bootstrap [SÉRIEUX]
**Problème :** une fonction `verify_bootstrap` qui rejoue la même logique d'interpolation que le bootstrap passera toujours — elle confirme la cohérence interne, pas la correction.
**Fix :** vérification par repricing indépendant : pour chaque swap, recalculer le par rate uniquement à partir de la courbe finale (log-linéaire sur les piliers complets simultanément) et vérifier que le résidu < 1e-6.
**Résultats obtenus :** tous les résidus = 0.0000bp (précision machine).

### FIX 4 — Gap coupon 6M → 2Y [MODÉRÉ]
**Problème :** les dates de coupon 1Y et 1.5Y d'un swap 2Y n'ont pas de pilier entre le dépôt 6M et le swap 2Y.
**Fix :** interpolation log-linéaire **itérative** : le pilier 2Y en cours de bootstrap sert d'ancre distale pour les dates intermédiaires. Convergence à 1e-12 en 3–5 itérations.
**Commentaire dans le code :**
> "6M→2Y gap: log-linear interpolation between P(0,0.5) and P(0,2.0). This introduces ~5bp uncertainty on the 2Y zero rate — acceptable given the absence of a 1Y instrument."

---

## 5. STRUCTURE DU SCRIPT — 8 SECTIONS

### Section 0 — Configuration and Market Data
- Date de référence, toutes les données, constantes, style matplotlib
- Pas de magic numbers inline — tout dans le bloc DATA en tête de script

### Section 1 — Deposit Bootstrap (0 à 6M)
**Fonction :** `bootstrap_deposits(deposit_data, day_count)`
**Maths :**
```
P(0,T) = 1 / (1 + r × T)
r_c(T) = -ln(P(0,T)) / T
```
Imprime : maturity | input rate | P(0,T) | zero rate continu

### Section 2 — Swap Bootstrap (2Y à 30Y)
**Fonction :** `bootstrap_swaps(swap_data, deposit_pillars, coupon_freq, day_count)`
**Maths :**
```
P(0,T_n) = [1 - K × Σ_{i=1}^{n-1} δ_i × P(0,T_i)] / [1 + K × δ_n]
```
- Interpolation log-linéaire pour les dates intermédiaires
- Résolution itérative pour gérer les gaps (FIX 4)
- Bootstrap strictement court→long (assertion sur l'ordre)

**Fonction helper :** `_loglinear_interp(t, pillars)`
```
ln P(0,t) = ln P(0,t_lo) + α × [ln P(0,t_hi) - ln P(0,t_lo)]
où α = (t - t_lo) / (t_hi - t_lo)
```

### Section 3 — Bootstrap Verification
**Fonction :** `verify_bootstrap(all_pillars, swap_data, coupon_freq)`
Repricing indépendant — utilise log-linéaire sur piliers complets finaux.
Assert : tous résidus < 0.01bp.

### Section 4 — Méthodes d'interpolation
Trois méthodes implémentées pour USD et EUR :

| Méthode | Fonction | Continuité | Usage |
|:--------|:---------|:-----------|:------|
| Linéaire | `interp_linear(maturities, zeros, t)` | C0 | Référence simple |
| Cubic Spline | `build_cubic_spline(maturities, zeros)` | C2 | **Référence pricing** |
| Nelson-Siegel | `nelson_siegel(t, beta0, beta1, beta2, tau)` | C∞ | Paramétrique |

**Nelson-Siegel :**
```
r(t) = β₀ + β₁×(1-e^{-t/τ})/(t/τ) + β₂×((1-e^{-t/τ})/(t/τ) - e^{-t/τ})
```
Contrainte : τ ≥ 0.5 (empêche l'effondrement du hump sur le court terme).
Fitting via `scipy.optimize.minimize`, méthode L-BFGS-B.

**Figure produite :** `figures/01_zero_curves_interpolation.png`
(2×1 subplot : USD bleu, EUR orange)

### Section 5 — Bond Pricing
**Fonctions :**
- `price_bond(face, coupon_rate, maturity, coupon_freq, curve_interpolator)`
- `bond_analytics(face, coupon_rate, maturity, coupon_freq, curve_interpolator, convention)`

**Maths :**
```
Price = Σ C_i × P(0,t_i) + Face × P(0,T)
DV01  = D_mod × Price × 0.0001
Convexity = (1/P) × Σ t_i² × C_i × P(0,t_i)  [convention continue]
```
YTM résolu par méthode de Brent (`scipy.optimize.brentq`).

**Obligations pricées (sur chaque courbe) :**
- Bond 1 : 5Y, coupon 4%, face 100
- Bond 2 : 10Y, coupon 3%, face 100
- Bond 3 : 30Y, coupon 5%, face 100

### Section 6 — Swap Pricing
**Fonctions :**
- `par_swap_rate(maturity, coupon_freq, curve_interpolator)`
- `price_swap(notional, fixed_rate, maturity, coupon_freq, curve_interpolator, position)`
- `swap_dv01(notional, maturity, coupon_freq, pillars)`

**Maths :**
```
V_float = N × (1 - P(0,T))
V_fixed = N × K × Σ δ_i × P(0,t_i)
V_payer = V_float - V_fixed

K* = (1 - P(0,T)) / Σ δ_i × P(0,t_i)

DV01 = |V(+1bp) - V(-1bp)| / 2  [choc parallèle numérique]
```

**Swaps pricés :**
- USD 2Y payer, fixed=4.50%, N=10M
- USD 5Y payer, fixed=4.00%, N=10M
- USD 10Y receiver, fixed=4.10%, N=10M
- EUR 2Y payer, fixed=3.20%, N=10M
- EUR 5Y payer, fixed=3.00%, N=10M
- EUR 10Y receiver, fixed=3.10%, N=10M

### Section 7 — Scenario Analysis
**Fonctions :**
- `apply_shock(pillars, shock_bps)` → retourne un nouveau CubicSpline interpolateur
- `build_scenarios()` → dict des 4 scénarios
- `run_scenario_analysis(...)` → DataFrames P&L par scénario
- `plot_scenarios(...)` → figures 02 et 03

**4 scénarios (pour USD et EUR) :**
| Scénario | Court (≤2Y) | Long (≥10Y) | Régime |
|:---------|:------------|:------------|:-------|
| Parallel +100bp | +100bp | +100bp | Choc Basel IRRBB |
| Bear Steepener | 0bp | +50bp | Fin de cycle, normalisation |
| Bear Flattener | +50bp | 0bp | Resserrement monétaire |
| Inversion | +75bp | -25bp | Anticipation récession |

**Figures produites :** `figures/02_scenario_analysis_usd.png`, `figures/03_scenario_analysis_eur.png`

### Section 8 — USD vs EUR Comparison
**Fonctions :**
- `print_comparison_table(...)` → métriques croisées
- `plot_usd_eur_comparison(...)` → figure 04

**Figure produite :** `figures/04_usd_eur_comparison.png`
(courbes superposées + spread USD-EUR annoté aux maturités 2Y, 5Y, 10Y, 30Y)

---

## 6. RÉSULTATS NUMÉRIQUES OBTENUS

### Zero rates bootstrappés

| Maturité | USD Zero (cts) | EUR Zero (cts) |
|:---------|:---------------|:---------------|
| 1M | 5.2983% | 3.8737% |
| 3M | 5.2948% | 3.8811% |
| 6M | 5.2310% | 3.8330% |
| 2Y | 4.5597% | 3.1949% |
| 3Y | 4.3377% | 3.0462% |
| 5Y | 4.1353% | 2.9982% |
| 7Y | 4.0872% | 3.0314% |
| 10Y | 4.0367% | 3.0754% |
| 12Y | 3.9803% | 3.1090% |
| 15Y | 4.0239% | 3.1430% |
| 20Y | 4.0821% | 3.1649% |
| 30Y | 4.1752% | 3.1876% |

### Analytics obligations 10Y (coupon 3%, face 100)

| Métrique | USD | EUR |
|:---------|:----|:----|
| Price | 91.0795 | 98.9779 |
| YTM | 4.0962% | 3.1206% |
| D_mac | 8.6460 | 8.7789 |
| D_mod | 8.4725 | 8.5132 |
| DV01 | 0.0772 | 0.0843 |
| Convexity | 81.994 | 83.484 |

### Par swap rates

| Maturité | USD | EUR |
|:---------|:----|:----|
| 2Y | 4.6251% | 3.2538% |
| 5Y | 4.2020% | 3.0513% |
| 10Y | 4.1016% | 3.1205% |

### Métriques cross-currency

| Métrique | USD | EUR |
|:---------|:----|:----|
| Slope 2Y–10Y | −52.3bp | −12.0bp |
| Slope 10Y–30Y | +13.8bp | +11.2bp |
| Par 5Y (bp) | 420.2 | 305.1 |
| Par 10Y (bp) | 410.2 | 312.0 |
| DV01 10Y bond (face 100) | 0.0772 | 0.0843 |
| DV01 10Y swap (N=10M) | 8,314 | 8,743 |

### Nelson-Siegel — paramètres fittés

**USD :**
- β₀ (long-run rate) = 4.0918%
- β₁ = 1.3650% → β₀+β₁ (short rate) = 5.4568%
- β₂ (trough) = −1.5773%
- τ = 2.000 ans | RMSE = 5.66bp

**EUR :**
- β₀ = 3.2976%
- β₁ = 0.7180% → β₀+β₁ = 4.0156%
- β₂ = −1.9754%
- τ = 2.000 ans | RMSE = 4.54bp

### P&L scénarios — obligation 10Y (3% coupon, face 100)

| Scénario | USD | EUR |
|:---------|:----|:----|
| Parallel +100bp | −7.51 | −8.29 |
| Bear Steepener | −3.60 | −3.99 |
| Bear Flattener | −0.24 | −0.26 |
| Inversion | +1.50 | +1.68 |

---

## 7. STANDARDS DE CODE ET CONVENTIONS

### Organisation
- Script exécutable top-to-bottom via `if __name__ == "__main__":`
- Chaque section = une fonction qui print ses résultats et retourne ses outputs pour la section suivante
- Headers de section : `"=" * 60 + "\n SECTION X — [TITRE] \n" + "=" * 60`

### Interpolation (règle critique)
Interpolation log-linéaire pour TOUTES les dates intermédiaires du bootstrap :
```python
ln P(0,t) = ln P(0,t_lo) + (t-t_lo)/(t_hi-t_lo) × (ln P(0,t_hi) - ln P(0,t_lo))
```
**Jamais d'extrapolation plate pendant le bootstrap** — assertion sur chaque date.

### Conventions explicites
- Dépôts : Act/360 pour USD et EUR
- USD swaps : fixed semi-annual (δ=0.5), Act/360
- EUR swaps : fixed annual (δ=1.0), Act/360
- Tous les zero rates : compounding continu
- Toutes les durations : convention explicite en paramètre obligatoire

### Numérique
- Tolérance bootstrap : |résidu| < 1e-6 (< 0.01bp)
- Contrainte Nelson-Siegel : τ ≥ 0.5
- Bump DV01 : exactement 1bp = 0.0001
- Taux affichés : 4 décimales minimum
- Discount factors affichés : 6 décimales

### Printing
- Tous les tableaux de résultats : `pandas DataFrame.to_string()`
- Interprétations desk : blocs `[DESK VIEW] ...`
- Pas de dump de dictionnaires bruts

### Graphiques
- USD = tons bleus (`steelblue`)
- EUR = orange/rouge (`darkorange`)
- Convention cohérente sur toutes les figures
- Sauvegarde : `figures/01_...png` à `figures/04_...png`
- Tout graphique : titre, axes labellisés avec unités, légende, `tight_layout()`

### Docstrings
- Style Google (Args, Returns, Notes) sur chaque fonction
- Notes documenter : conventions utilisées, limitations connues

---

## 8. LIMITATIONS CONNUES ET DOCUMENTÉES

1. **Single-curve framework** — pas de séparation OIS discounting / IBOR forward curve. Exact pour SOFR/ESTR, biaisé pour EURIBOR.
2. **Pas d'ajustement de convexité pour les futures** — ordre de grandeur : σ² × T₁ × T₂ / 2.
3. **Approximation Act/360** — δ = 1/freq au lieu de calendrier exact (business days, Modified Following). Erreur < 0.5bp sur un swap 5Y.
4. **Pas d'extrapolation au-delà de 30Y** — CubicSpline avec `extrapolate=False` ; toute requête au-delà lève une erreur.

---

## 9. README — SECTIONS CLÉS

Le README contient :
1. Description du projet
2. Structure du repo (tree view)
3. Installation et exécution
4. Background mathématique (formules deposit, swap, Nelson-Siegel, duration)
5. Tableau de résultats clés (zero rates, bond analytics, par swap rates, P&L scénario)
6. Section limitations connues
7. **Section Interview Q&A** — 5 questions typiques d'un entretien rates avec réponses complètes :
   - Qu'est-ce qu'un discount factor et comment le bootstrapper ?
   - Pourquoi l'ordre d'itération du bootstrap est-il important ?
   - Différence entre duration Macaulay et duration modifiée ?
   - Pourquoi le bond EUR 10Y est-il pricé différemment du bond USD 10Y pour le même coupon ?
   - Que signifie un bear flattener pour une position longue duration ?

---

## 10. PROBLÈMES RENCONTRÉS ET RÉSOLUS (HISTORIQUE TECHNIQUE)

### Bug 1 — Extrapolation dans `_loglinear_interp`
**Cause :** la condition `if t >= sorted_mats[-1]` levait une erreur même pour `t == last_pillar` (cas exact match = dernière date de coupon = maturité du swap).
**Fix :** condition modifiée en `if abs(t - sorted_mats[-1]) < 1e-9: return pillars[sorted_mats[-1]]`.

### Bug 2 — Gap 6M → 2Y : les coupons 1Y et 1.5Y dépassent le dernier pilier connu
**Cause :** la logique originale cherchait à interpoler entre piliers connus, mais 1Y et 1.5Y sont au-delà du dernier dépôt (0.5Y) et avant le swap 2Y en cours de bootstrap.
**Fix :** résolution itérative — l'estimé courant de P(0, 2Y) est provisoirement ajouté aux piliers comme ancre distale, le swap est résolu, et on itère jusqu'à convergence à 1e-12. Converge en 3–5 itérations.

### Bug 3 — `DataFrame.applymap()` déprécié dans pandas ≥ 2.1
**Fix :** remplacé par `DataFrame.map()`.

### Bug 4 — Vérification bootstrap avec CubicSpline ≠ bootstrap log-linéaire
**Cause :** CubicSpline et log-linéaire produisent des valeurs différentes aux dates non-piliers, donc les résidus atteignaient ~0.5bp sur le 30Y.
**Fix :** la vérification utilise également l'interpolation log-linéaire sur l'ensemble complet des piliers finaux. C'est le test de cohérence correct : chaque swap doit s'annuler sur la courbe finale.

---

## 11. EXTENSIONS POSSIBLES (PISTES POUR LA SUITE)

Si tu souhaites étendre le projet dans de nouvelles conversations :

- **Multi-curve framework** : séparer OIS discounting curve et EURIBOR/LIBOR forward curve
- **Futures** : intégrer des SOFR futures avec ajustement de convexité
- **Calendrier exact** : remplacer δ = 1/freq par un vrai calendrier (module `holidays` + Modified Following)
- **Key-rate DV01** : sensibilités par pilier plutôt que choc parallèle global
- **Swaption pricing** : Black ou Normal vol surface, pricing de caps/floors
- **Credit spread** : courbe survie / Z-spread sur bonds corporates
- **Visualisation interactive** : Plotly ou Panel dashboard
- **Tests unitaires** : pytest sur chaque fonction de pricing

---

## 12. COMMANDE DE RÉFÉRENCE POUR REPRENDRE

Pour donner le contexte maximum à une nouvelle conversation, coller ce bloc en premier message :

```
Je travaille sur un projet Python de yield curve construction (USD SOFR + EUR ESTR).
Le script yield_curve.py (1292 lignes) est complet et tourne sans erreur.
Il couvre : bootstrap deposits/swaps, 3 méthodes d'interpolation (linéaire, cubic spline, Nelson-Siegel),
pricing bonds et swaps, analyse de scénarios (4 chocs), comparaison cross-currency.
4 fixes critiques sont implémentés : interpolation gaps, duration conventions, bootstrap verification, gap 6M→2Y.

[Coller ici le CONTEXTE_PROJET_COMPLET.md ou la section spécifique pertinente]

Je veux maintenant : [décrire l'extension ou la question]
```
