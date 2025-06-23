
# Personalized Risk Stratification for Covid Vaccine Adverse Events

This repository contains data and code scripts for association rule mining pipelines for analyzing adverse drug reactions (ADRs) related to covid vaccines. The project is focused on refining ADR signals based on various contributing factors such as medications, health conditions, and demographics like age, sex, and dose history and also, based on the association strengths specifically, clinical significance.

## Repository Structure

```
📦Personalized-Risk-Stratification
 ┣ 📂Data/
 ┃ ┣━ Final Datasets
 ┃    ┗━ CSV files with final data
 ┃ ┣━ Target Label
 ┃    ┗━ XLSX files with patient symptom and label data
 ┣ 📂Code/
 ┃ ┣━ data_preprocessing.ipynb
 ┃ ┣━ Moderna Pattern Analysis
 ┃    ┗━ .py files of 5 different associations.
 ┃ ┣━ Pfizer Pattern Analysis
 ┃    ┗━ .py files of 5 different associations.
 ┣ 📄 requirements.txt
 ┗ 📄 README.md
```

## Key Modules

- `data_preprocessing.ipynb`: Cleans, filters, and prepares the input data for pattern analysis.
- `arm_med_to_adr.py`: Mines association rules between extracted medications and ADRs.
- `arm_hc_to_adr.py`: Extracts associations between pre-existing health conditions and ADRs.
- `arm_hc_med_to_adr.py`: Identifies strong rules using both health conditions and medications.
- `arm_age_sex_dose_to_adr.py`: Explores demographic and dose factors leading to ADRs.
- `arm_age_sex_hc_dose_to_adr.py`: Combines demographics, dose, and health conditions to find ADR risks.

## Outputs

To access all the results for the association pattern analysis visit - link

## Set-up

### 1. Clone the Repository

```bash
git clone https://github.com/SaiSharanyaY/Personalized-Risk-Stratification.git
cd Personalized-Risk-Stratification
```

### 2. Install Dependencies

Use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Run the code

- Each patient record is analyzed across multiple dimensions to ensure robust signal detection.

## Acknowledgment

This work is validated and cross-checked by clinical domain experts.

## License

This repository is intended for academic research use only. Contact the author for other uses.



