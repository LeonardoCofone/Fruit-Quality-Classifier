# Fruits and Vegetables Quality Classifier  
## MobileNetV2 + Transfer Learning on a 5 GB Custom Image Dataset  

---

## 1 · Introduzione

Questo repository contiene il codice completo e la documentazione per un sistema di classificazione automatica dello stato **fresh / rotten** di 28 diverse tipologie di frutta e verdura.  
Il progetto è stato interamente realizzato da Leonardo Cofone: dalla raccolta e pulizia dati, fino all’addestramento di un modello deep‑learning e alla definizione di uno script di inferenza pronto all’uso.

---

## 2 · Dataset

| Caratteristica | Valore |
|----------------|--------|
| Origine        | Raccolta e annotazione proprietaria |
| Dimensione     | ≈ 5 GB complessivi (immagini `.jpg`) |
| Classi totali  | 56 (28 prodotti × 2 condizioni: `fresh`, `rotten`) |
| Risoluzione    | Variabile; ridimensionata a **128 × 128** nel preprocessing |

Il dataset è organizzato in cartelle:  
Unified_Dataset/
└── <fruit_name>/
├── fresh/
│ └── *.jpg
└── rotten/
└── *.jpg

> **Nota** Il dataset non è caricato su GitHub per ragioni di spazio; è disponibile come file `kaggle/input/fruitquality1/Unified_Dataset` o su richiesta.

---

## 3 · Dipendenze principali

| Pacchetto                | Versione testata |
|--------------------------|------------------|
| Python                   | 3.11 |
| TensorFlow / Keras       | 2.16 |
| scikit‑learn             | 1.6.1 |
| imbalanced‑learn         | 0.13 |
| Pillow                   | 10 |
| seaborn / matplotlib     | latest |

Installazione rapida:

```bash
pip install -r requirements.txt

