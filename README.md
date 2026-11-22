# Leyenda : tri, débruitage et légendage automatique de photos

Projet de data science réalisé en groupe à CESI École d'Ingénieurs (majeure Data Science et IA), pour un client fictif : **TouNum**, une entreprise qui numérise de grandes collections d'images pour ses clients.

TouNum veut proposer un nouveau service : générer automatiquement une légende pour chaque photo numérisée, afin de rendre les fonds d'archives faciles à rechercher. Le problème, c'est que les fonds sont hétérogènes (photos, schémas, textes scannés, dessins) et souvent abîmés par la numérisation.

La solution est une chaîne en trois étapes, une par notebook.

| Étape | Notebook | Objectif | Résultat principal |
|:---|:---|:---|:---|
| 1 | [Classification des images](notebooks/1ClassificationPhotos.ipynb) | Garder uniquement les photos | Accuracy test de 92,4 %, AUC 0,981, rappel photo réglé par seuil |
| 2 | [Débruitage](notebooks/2DebruitageAutoencodeur.ipynb) | Nettoyer les photos retenues | PSNR test moyen de 35,9 dB et SSIM de 0,934 |
| 3 | [Légendage](notebooks/3LegendageCnnRnn.ipynb) | Générer une phrase qui décrit la photo | CIDEr de 98,6 sur 5 000 photos de test, soit 112 % du score d'une légende humaine |

## 1. Classification : photo ou autre image

* Jeu de données découpé en entraînement (70 %), validation (15 %) et test (15 %).
* Cinq modèles CNN comparés, du plus simple au plus régularisé : trois modèles binaires (A, B, C) et deux modèles multiclasses à 5 catégories (D avec BatchNormalization, E avec GroupNormalization).
* Analyse biais et variance pour choisir le modèle, puis choix du seuil de décision sur la validation (seuil 0,345 : précision 92,8 %, rappel 96,9 %, F1 94,8 %).
* Test de robustesse sur des images de test dégradées de quatre façons, jamais vues à l'entraînement.
* Le modèle multiclasse sert aussi de filtre photo : F1 macro de 91,9 % sur le test.

## 2. Débruitage par autoencodeur convolutif

* Autoencodeur convolutif de référence, puis architecture UNet avec connexions de saut.
* Évaluation par PSNR et SSIM, détaillée par niveau de bruit.
* Le UNet améliore le PSNR de 6,7 dB en moyenne par rapport à l'autoencodeur simple (intervalle bootstrap à 95 % de 6,1 à 7,5 dB, test de Wilcoxon p = 2,4e-07).
* Étude d'ablation sur l'augmentation de données, puis test sur 68 vraies photos pour mesurer le gain réel par rapport à la photo non traitée.

## 3. Légendage : CNN encodeur et RNN décodeur

* Données MS COCO : 123 287 images et leurs légendes (113 287 en entraînement, 5 000 en validation, 5 000 en test).
* Encodeur InceptionV3 pré entraîné et gelé : chaque image devient 64 régions de 2 048 valeurs.
* Décodeur LSTM avec attention de Bahdanau, vocabulaire de 10 027 mots.
* Recherche par grille sur 10 réglages (dropout, lissage des étiquettes, régularisation de l'attention, taux d'apprentissage) : CIDEr de validation de 89,7 pour le modèle de base à 93,3 pour le modèle optimisé.
* Affinage par apprentissage par renforcement (SCST, self critical sequence training) qui optimise directement le CIDEr : de 93,3 à 101,6 en validation.
* Ensemble de modèles et règles d'écriture (pas de mot répété, pas de phrase coupée).
* Évaluation finale sur 5 000 photos de test avec BLEU 1 à 4, ROUGE L et CIDEr : CIDEr de 98,6, contre 7,0 pour une phrase constante et 53,1 pour la méthode du plus proche voisin.
* Score de confiance par légende : les 20 % de légendes les moins sûres sont envoyées en relecture humaine.

## Lancer les notebooks

Les notebooks ont été exécutés sur Google Colab avec un GPU T4. Les sorties (courbes, tableaux, exemples d'images) sont conservées pour pouvoir les lire directement sur GitHub.

```bash
pip install -r requirements.txt
jupyter notebook
```

Les jeux de données ne sont pas inclus dans le dépôt (trop volumineux). Chaque notebook indique au début où les placer.

## Technologies

Python, TensorFlow et Keras, NumPy, Pandas, Scikit Learn, Matplotlib, scikit image, Google Colab.

## Auteur

Nizar, élève ingénieur en Data Science et IA à CESI.
[LinkedIn](https://www.linkedin.com/in/nizar-outski-549941383/)
