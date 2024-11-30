# **Projet Deep Learning - Classification Multi-Modale**

Ce projet a été réalisé dans le cadre du cours de **Deep Learning** en ING 5 - Data & IA à l'**ECE Paris** par **Mathéo GONNET** et **Ines GOULAMHOUSSEN**.

---

## **Description Générale**

L'objectif de ce projet est de développer un pipeline multi-modal capable de traiter des données d'images et de texte en utilisant des techniques avancées de deep learning. Le projet se décompose en plusieurs étapes clés :

1. **Classification d'images** à l'aide d'un réseau neuronal convolutionnel (CNN).
2. **Analyse des descriptions textuelles** associées aux images via des embeddings de mots et un réseau neuronal récurrent (RNN).
3. **Fusion des informations visuelles et textuelles** pour améliorer les performances grâce à une approche multi-modale.
4. **Test du modèle fusionné** sur des images personnalisées.

---

## **Structure du Projet**

```
multi_modal_project/
│
├── datasets/
│   └── flickr8k/
│       ├── images/
│       └── captions.txt
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_image_model_training.ipynb
│   ├── 03_text_model_training.ipynb
│   ├── 04_fusion_model_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_random_pict_testing.ipynb
│
├── models/
│   ├── cnn_model.h5
│   ├── rnn_model.h5
│   └── fusion_model.h5
│
└── requirements.txt

```
Le projet est organisé en plusieurs notebooks Jupyter, chacun correspondant à une étape spécifique du pipeline :

### **Notebook 1 : Prétraitement des Données**

- **Description** : Ce notebook prépare les données pour l'entraînement des modèles. Il traite les images et les légendes associées, effectue la tokenisation du texte, le padding des séquences, et encode les labels pour la classification multi-label.
- **Pourquoi** : Un prétraitement adéquat est essentiel pour assurer que les données sont dans un format approprié pour les modèles de deep learning. Cela permet d'améliorer l'efficacité et les performances des modèles.

### **Notebook 2 : Entraînement du Modèle CNN**

- **Description** : Ce notebook entraîne un modèle CNN pour la classification des images. Le modèle utilisé est **MobileNetV2** pré-entraîné sur ImageNet, avec des couches supplémentaires pour adapter le modèle à notre jeu de données.
- **Pourquoi MobileNetV2** : MobileNetV2 est un modèle léger et efficace, idéal pour être utilisé comme extracteur de caractéristiques. Il offre un bon compromis entre performance et complexité computationnelle.
- **Étapes Clés** :
  - Chargement et prétraitement des images.
  - Utilisation de **tf.data.Dataset** pour gérer efficacement les données en mémoire.
  - Entraînement du modèle avec fine-tuning des dernières couches pour améliorer les performances.

### **Notebook 3 : Entraînement du Modèle RNN**

- **Description** : Ce notebook entraîne un modèle RNN pour analyser les descriptions textuelles. Un **LSTM** est utilisé pour capturer les dépendances séquentielles dans les légendes.
- **Pourquoi LSTM** : Les LSTM sont efficaces pour modéliser les données séquentielles et peuvent gérer les dépendances à long terme dans le texte.
- **Étapes Clés** :
  - Conversion des légendes en séquences numériques.
  - Utilisation d'une couche **Embedding** pour représenter les mots en vecteurs.
  - Entraînement du modèle LSTM pour prédire les labels associés aux descriptions.

### **Notebook 4 : Fusion Multi-Modale**

- **Description** : Ce notebook fusionne les modèles CNN et RNN pour créer un modèle multi-modal qui prend en entrée à la fois l'image et la description textuelle.
- **Comment** :
  - Extraction des caractéristiques visuelles et textuelles des modèles pré-entraînés.
  - Fusion des caractéristiques via une couche de **concaténation**.
  - Ajout de couches fully connected pour apprendre les interactions entre les modalités.
- **Pourquoi la Fusion** : La fusion des informations visuelles et textuelles permet au modèle de capturer des relations plus complexes et d'améliorer les performances de classification.

### **Notebook 5 : Évaluation du Modèle**

- **Description** : Ce notebook évalue les performances du modèle fusionné sur un ensemble de test. Il génère des métriques telles que la précision et la courbe ROC.
- **Pourquoi** : L'évaluation est cruciale pour comprendre les performances du modèle et identifier les zones d'amélioration.

### **Notebook 6 : Test du Modèle avec des Images Personnalisées**

- **Description** : Ce notebook permet de tester le modèle fusionné sur des images personnalisées et d'afficher les prédictions.
- **Comment Tester avec Vos Propres Images** :
  1. **Préparation de l'Image** :
     - Placez vos images dans le répertoire approprié et mettez à jour le chemin dans le code.
     - Assurez-vous que les images sont au format JPEG ou PNG.
  2. **Prétraitement** :
     - L'image sera automatiquement redimensionnée et normalisée lors du chargement.
  3. **Légende Associée** :
     - Vous pouvez fournir une légende personnalisée pour accompagner l'image.
     - La légende sera nettoyée et convertie en séquence pour être utilisée par le modèle.
  4. **Exécution du Notebook** :
     - Lancez le notebook et exécutez les cellules pour obtenir les prédictions.
     - Les classes prédites seront affichées avec leurs probabilités associées.
- **Affichage des Résultats** :
  - Le notebook affiche l'image avec les classes prédites en titre.
  - Un graphique en barres montre les probabilités des classes prédites.

---

## **Comment Utiliser le Projet**

1. **Cloner le Répertoire** :
   - Récupérez l'ensemble des notebooks et des fichiers associés.

2. **Installer les Dépendances** :
   - Assurez-vous d'avoir installé les bibliothèques nécessaires, notamment TensorFlow, Keras, NumPy, et Matplotlib.

3. **Organisation des Données** :
   - Placez les données du jeu de données (images et annotations) dans le répertoire `../datasets/flickr8k/` ou ajustez les chemins dans les notebooks.

4. **Exécuter les Notebooks dans l'Ordre** :
   - Commencez par le Notebook 1 pour le prétraitement.
   - Enchaînez avec les Notebooks 2, 3, et 4 pour l'entraînement des modèles.
   - Utilisez le Notebook 5 pour évaluer les performances.
   - Enfin, testez le modèle avec vos propres images dans le Notebook 6.

5. **Tester avec Vos Propres Images** :
   - Dans le Notebook 6, modifiez la liste `test_image_paths` pour inclure le chemin vers vos images.
     ```python
     test_image_paths = ["../chemin/vers/votre_image1.jpg", "../chemin/vers/votre_image2.png"]
     ```
   - Si vous souhaitez fournir une légende personnalisée, modifiez la variable `test_caption` :
     ```python
     test_caption = "Votre légende ici"
     ```
   - Exécutez le notebook pour voir les prédictions du modèle sur vos images.

---

## **Conclusion**

Ce projet démontre l'efficacité d'une approche multi-modale pour la classification d'images en combinant des techniques de deep learning pour le traitement d'images et de texte. Grâce à la fusion des caractéristiques visuelles et textuelles, le modèle est capable de capturer des informations riches et d'améliorer les performances de classification.

---

**Mathéo GONNET & Ines GOULAMHOUSSEN**

**ECE Paris - ING 5 Data & IA**







