# MRI-Image-Classification-for-Brain-Tumor-Detection-

Ovaj repozitorij sadrži kod i bilježnice za detekciju tumora mozga na MRI slikama korištenjem različitih metoda strojnog učenja i dubokog učenja (CNN, ResNet50, KNN, SVM, VGG).

## Sadržaj

- `notebooks/` – Jupyter bilježnice s treniranjem modela
- `data/` – Struktura podataka
- `src/` – Dodatni pomoćni kod (skripte)
- `README.md` – Ovaj dokument

## Preduvjeti

- Python 3.8+
- Preporučuje se korištenje virtualnog okruženja (`venv` ili `conda`)
- Preporučene biblioteke:
  - numpy
  - matplotlib
  - opencv-python
  - scikit-learn
  - tensorflow / keras
  - imutils
  - seaborn
  - scikit-image

Instalacija svih potrebnih paketa:
```sh
pip install -r requirements.txt
```

## Priprema podataka 


1. Preuzmite dataset [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data) i raspakirajte ga u `data/BrainTumorMRIDataset/`.


## Korištenje

## Korištenje

### 1. Klonirajte repozitorij

```sh
git clone https://github.com/korisnicko_ime/MRI-Image-Classification-for-Brain-Tumor-Detection-.git
cd MRI-Image-Classification-for-Brain-Tumor-Detection-
```

### 2. Aktivirajte virtualno okruženje

```sh
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 3. Instalirajte ovisnosti

```sh
pip install -r requirements.txt
```

### 4. Pokrenite Jupyter Notebook

```sh
jupyter notebook
```
ili otvorite `.ipynb` datoteke u Visual Studio Code.

### 5. Priprema podataka

1. Pomoću skripte `merging_scipt.py` iz `/src` pretvorite podatke iz `data/BrainTumorMRIDataset/` u `data/BinaryBrainTumorDataset/`
2. Provjerite da struktura direktorija odgovara putanjama u bilježnicama (npr. `data/BinaryBrainTumorDataset/Training/yes`, `no`).

### 6. Pokretanje treniranja modela

- Otvorite željenu bilježnicu iz `notebooks/` (npr. `training_resnet.ipynb`, `training_5fold.ipynb`, `training_knn_hog_glcm.ipynb`, `training_svm_hog_glcm.ipynb`)
- Prilagodite putanje do podataka ako je potrebno
- Pokrenite ćelije redom

## Bilješke

- Za treniranje dubokih modela preporučuje se korištenje GPU-a.
- 5-fold cross-validation može biti vrlo zahtjevna za računalo.
- Modeli i rezultati se spremaju u `notebooks/` direktorij.

## Kontakt

Za pitanja i prijedloge otvorite [issue](https://github.com/korisnicko_ime/MRI-Image-Classification-for-Brain-Tumor-Detection-/issues) ili pošaljite email na [jrubcic@uniri.hr](mailto:jrubcic@uniri.hr).

---

**Napomena:** Podaci nisu uključeni u repozitorij zbog veličine i autorskih prava. Pratite upute za preuzimanje podataka.
