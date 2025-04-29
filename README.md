# Parkinson Detection API 🧠🔬

This repository provides a simple REST API to predict Parkinson's disease:
- Either from **numerical clinical data** (CSV/form)
- Or from a **spiral drawing image**

## 🚀 Endpoints

### 🔹 `POST /predict`
For structured clinical input.

**JSON Input:**
```json
{
  "UPDRS": 67.83,
  "FunctionalAssessment": 2.13,
  "MoCA": 29.92,
  "Tremor": 1,
  "Rigidity": 0,
  "Bradykinesia": 0,
  "Age": 70,
  "AlcoholConsumption": 2.24,
  "BMI": 15.36,
  "SleepQuality": 9.93,
  "DietQuality": 6.49,
  "CholesterolTriglycerides": 395.66
}
