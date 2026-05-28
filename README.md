# KinetiX-LDS : Latent Dynamics Sensor for LLM Regime Transitions

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![CUDA Supported](https://img.shields.io/badge/CUDA-Supported-green.svg)
![Status: Experimental](https://img.shields.io/badge/Status-Experimental-orange.svg)

**KinetiX-LDS** est un capteur d'état matériel à ultra-basse latence (sub-100 µs) qui surveille en temps réel les transitions de régime de représentation au sein des couches intermédiaires d'un LLM afin d'aiguiller dynamiquement les requêtes entre des modèles locaux légers (Edge) et des API propriétaires haut de gamme (Cloud).

---

### ⚡ Chiffres Clés & ROI (Lecture 30 secondes)

*   **77 % d'Économie Cloud** : Réduction immédiate des coûts d'inférence en absorbant le trafic nominal sur site et en n'escaladant que les requêtes complexes vers le Cloud.
*   **8.4 µs d'Overhead** : Évaluation asynchrone ultra-rapide via des hooks C++ binarisés (HDC), évitant le calcul coûteux d'une entropie finale sur vocabulaire (Shannon) qui pénalise de près de 19 ms par token.
*   **Intégration en 3 Lignes de Code** : Se connecte sur n'importe quel LLM existant (Llama, Qwen, SmolLM) sans réentraînement, ni fine-tuning.



### 🛠️ Intégration Technique (Concrètement)

```python
from kinetix import KinetiXStreamMonitor

# 1. Instancier le moniteur à double échelle de temps
monitor = KinetiXStreamMonitor(dim_size=2048, alpha=0.20, beta=0.85, threshold=0.132)

# 2. Enregistrer le hook PyTorch sur une couche médiane du LLM local (ex: couche 12 de Qwen)
model.model.layers[12].register_forward_hook(monitor.pytorch_hook)

# 3. Intercepter le drift (8.4 µs en C++) et escalader vers le Cloud si Drift > Seuil
```

---

## 🛑 Ce que KinetiX-LDS ne fait PAS
Pour prévenir tout malentendu sur ce projet de recherche en interprétabilité des représentations :
*   Il **ne comprend pas** la sémantique et **ne remplace pas** les guardrails de sécurité morale/éthique.
*   Il **ne détecte pas** l'intention malveillante de manière isolée (les intentions inoffensives complexes comme du code Python légitime et les attaques par suffixe GCG provoquent une déformation similaire de l'espace latent).
*   Il **mesure strictement** la transition vers un régime à haute contrainte syntaxique ou structurelle.

## 📉 Constats Négatifs & Rigueur Scientifique
Nos tests d'isolation rigoureux (N=400) ont mis en évidence que :
*   L'intention malveillante (ex: jailbreak) et la syntaxe complexe bénigne (ex: JSON imbriqué, LaTeX, code) sont **indiscernables** géométriquement dans les couches cachées.
*   La dérive détectée par KinetiX-LDS est donc de nature **structurelle** (OOD / transition de régime) et non morale.

## ⚠️ Limites Connues
*   **Calibrage dépendant de l'architecture** : Les seuils de drift doivent être calibrés individuellement pour chaque modèle.
*   **Binarisation HDC** : La projection Charikar (SimHash) binarisée troque une légère précision géométrique contre une vitesse d'évaluation sub-microseconde.

## 📂 Structure du Dépôt
*   `kinetix_core.py` : Le SDK unifié (Router MLOps + Stream Monitor).
*   `kinetix_production_stress.py` : Script de stress test standard local et évaluation de flux.
*   `LICENSE` : Licence AGPLv3.
*   `README.md` : Ce fichier de présentation.

## 📄 Licence & Citation
Ce projet est distribué sous licence **GNU Affero General Public License v3 (AGPLv3)**.
```bibtex
@misc{kinetix2026,
  title={KinetiX-LDS: Representational Regime Transition Detection in LLM Latent Space},
  author={JohnDoerch-Eng},
  year={2026},
  howpublished={GitHub repository},
}
```
