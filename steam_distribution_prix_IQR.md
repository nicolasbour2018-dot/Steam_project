# Distribution des prix Steam et identification des valeurs atypiques

## Objectif

Nous cherchons à visualiser la distribution des prix des jeux Steam tout en distinguant :

- les **prix courants** au regard de la distribution statistique ;
- les **prix statistiquement atypiques** (*outliers*).

Dans le notebook, nous travaillons sur le DataFrame `clean_df` et sur la variable `price`, déjà convertie en dollars.

> **Important :** un *outlier statistique* n'est pas nécessairement une donnée aberrante ou erronée.  
> Un jeu vendu 40 $ ou 60 $ peut être parfaitement cohérent d'un point de vue métier tout en étant rare dans ce dataset.

---

## 1. Définir mathématiquement les outliers avec l'IQR

Les quartiles sont des **valeurs de coupure** dans une série triée :

- `Q1` : valeur située à 25 % de la distribution ;
- `Q2` : médiane, située à 50 % ;
- `Q3` : valeur située à 75 %.

L'**écart interquartile** (*InterQuartile Range*, IQR) mesure la largeur de la zone contenant les 50 % centraux des observations :

\[
IQR = Q3 - Q1
\]

La règle classique de Tukey considère comme statistiquement atypiques les valeurs situées au-delà de :

\[
Upper\ Bound = Q3 + 1.5 \times IQR
\]

Le facteur `1.5` est une **convention statistique**.  
Ici, nous nous intéressons surtout à la borne supérieure puisque les prix sont positifs.

### PySpark

```python
# Calcul de Q1 et Q3 sur la variable price
q1, q3 = clean_df.approxQuantile(
    "price",
    [0.25, 0.75],
    0.001
)

# Écart interquartile
iqr = q3 - q1

# Seuil supérieur des valeurs atypiques
upper_bound = q3 + 1.5 * iqr

print(f"Q1 : {q1:.2f} $")
print(f"Q3 : {q3:.2f} $")
print(f"IQR : {iqr:.2f} $")
print(f"Upper bound : {upper_bound:.2f} $")
```

### Interprétation

L'`upper_bound` est donc le **seuil statistique** à partir duquel un prix sera considéré comme atypique dans **cette distribution**.

Ce seuil n'est pas modifié en fonction de notre connaissance du marché : l'analyse métier viendra ensuite compléter l'analyse statistique.

---

## 2. Identifier les prix statistiquement atypiques

Nous conservons les données et ajoutons simplement une variable catégorielle permettant de distinguer les deux groupes.

```python
from pyspark.sql import functions as F

price_df = (
    clean_df
    .select("name", "price")
    .filter(F.col("price").isNotNull())
    .withColumn(
        "price_status",
        F.when(
            F.col("price") > upper_bound,
            "Outlier statistique"
        ).otherwise("Prix courant")
    )
)
```

Mathématiquement :

\[
price \leq Upper\ Bound \Rightarrow Prix\ courant
\]

\[
price > Upper\ Bound \Rightarrow Outlier\ statistique
\]

Aucune observation n'est supprimée : nous ajoutons seulement une information permettant de colorer ensuite la distribution.

---

## 3. Regrouper les prix en classes

Un regroupement direct avec :

```python
clean_df.groupBy("price")
```

produit de très nombreux prix distincts (`0.99`, `1.49`, `1.99`, `2.49`, etc.).  
La visualisation devient alors très irrégulière et difficile à lire.

Nous regroupons donc les prix en **classes de 2 $** :

```python
bin_width = 2

price_df = price_df.withColumn(
    "price_bin",
    F.floor(F.col("price") / bin_width) * bin_width
)
```

Par exemple :

| Prix réel | Classe |
|---:|---:|
| 0.00 à 1.99 $ | 0 |
| 2.00 à 3.99 $ | 2 |
| 4.00 à 5.99 $ | 4 |
| 6.00 à 7.99 $ | 6 |

Le choix de `2 $` constitue ici un compromis : suffisamment large pour éviter l'effet de « peigne » des prix exacts, mais suffisamment fin pour conserver la forme de la distribution.

---

## 4. Compter les jeux par classe de prix

Nous comptons maintenant les jeux dans chaque classe en séparant les prix courants et les outliers.

```python
price_distribution = (
    price_df
    .groupBy("price_bin", "price_status")
    .agg(
        F.count("*").alias("game_nb")
    )
    .orderBy("price_bin")
)

display(price_distribution)
```

Nous obtenons donc une petite table de la forme :

| price_bin | price_status | game_nb |
|---:|---|---:|
| 0 | Prix courant | ... |
| 2 | Prix courant | ... |
| 4 | Prix courant | ... |
| ... | ... | ... |
| 24 | Outlier statistique | ... |

Tous les traitements importants ont jusqu'ici été réalisés en **PySpark**.

---

## 5. Construire la visualisation bicolore

La table étant maintenant agrégée et très petite, nous pouvons la convertir en Pandas uniquement pour construire le graphique avec Matplotlib.

```python
plot_df = (
    price_distribution
    .groupBy("price_bin")
    .pivot(
        "price_status",
        ["Prix courant", "Outlier statistique"]
    )
    .sum("game_nb")
    .fillna(0)
    .orderBy("price_bin")
    .toPandas()
)
```

### Histogramme par classes de prix

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))

# Prix situés sous le seuil IQR
plt.bar(
    plot_df["price_bin"],
    plot_df["Prix courant"],
    width=bin_width * 0.9,
    label="Prix courant"
)

# Prix situés au-dessus du seuil IQR
plt.bar(
    plot_df["price_bin"],
    plot_df["Outlier statistique"],
    width=bin_width * 0.9,
    bottom=plot_df["Prix courant"],
    label="Outlier statistique"
)

# Seuil statistique
plt.axvline(
    upper_bound,
    linestyle="--",
    linewidth=2,
    label=f"Upper bound = {upper_bound:.2f} $"
)

plt.xlabel("Prix ($)")
plt.ylabel("Nombre de jeux")
plt.title("Distribution des jeux Steam selon leur prix")

plt.legend()
plt.show()
```

---

## Lecture du graphique

Cette visualisation permet de répondre à deux questions différentes :

1. **Comment les prix sont-ils distribués ?**  
   Les classes de prix montrent où se concentre la majorité des jeux.

2. **À partir de quelle valeur les prix deviennent-ils statistiquement atypiques ?**  
   La ligne verticale représente l'`upper_bound` calculé avec la règle de l'IQR et les barres correspondantes apparaissent dans la catégorie `Outlier statistique`.

Il faut ensuite distinguer **analyse statistique** et **analyse métier** :

> Un prix peut être statistiquement atypique parce qu'il est rare dans le dataset sans être aberrant dans le marché du jeu vidéo.

Ainsi, l'IQR fournit un critère mathématique objectif pour décrire la distribution.  
L'interprétation métier permet ensuite d'expliquer pourquoi certains jeux légitimement plus chers se trouvent malgré tout dans la zone statistiquement atypique.

---

## Option : inspecter directement les jeux situés au-dessus du seuil

Pour compléter le graphique :

```python
outlier_games = (
    price_df
    .filter(F.col("price") > upper_bound)
    .select("name", "price")
    .orderBy(F.desc("price"))
)

display(outlier_games)
```

Cela permet de vérifier concrètement quels jeux constituent la partie atypique de la distribution sans les considérer automatiquement comme des erreurs.
