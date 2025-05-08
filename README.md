# Data Science Project - Predict Diabetes From Medical Records

Build a predictive model to identify the likelihood of diabetes in patients based on their medical records. By leveraging structured health data such as Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin and so on.

## Datasets
- [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [Frankfurt Hospital (Germany) Type 2 Diabetes Datase](https://www.kaggle.com/datasets/johndasilva/diabetes/data)
> Install those files, and put them into './data'

```
E.g.
data/
| - .gitkeep
| - Frankfurt_Hospital_diabetes.csv
| - Pima_Indians_diabetes.csv
```

## Environment
```bash
# Follow the template to commit
git config commit.template .gitmessage.txt

# Install needed packages
pip install -r requirements.txt

## Install this project in editable mode (requires setup.py)
pip install -e .
```

## Run
- Run PIMA dataset: `sh ./launch/run_pima.sh`
- Run Frankfurt dataset:`sh ./launch/run_frankfurt.sh`

---

## 📁 資料夾與檔案說明

### `configs/`
- 儲存模型設定、訓練參數等 `.yaml` 檔。
- 建議命名方式：`train.yaml`, `test.yaml` 等。

---

### `cores/`
- 專案核心邏輯封裝（**流程控制**類別與函數）。
- `trainer.py`: 封裝訓練邏輯的 `Trainer` 類別（如訓練迴圈、驗證、儲存模型）。
- `inferencer.py`: 封裝推論邏輯的 `Inferencer` 類別（例如單張預測、batch 推論等）。

---

### `data/`
- 可能使用到的資料集。
- 可放 `diabetes.csv` 等。

---

### `launch`
- 程式執行的 scripts。

---

### `outputs/`
- 訓練或推論結果的輸出資料夾。
- 子資料夾建議用途：
  - `tools/`: 推論可視化、結果分析腳本。
  - `training/`: 儲存訓練結果（如 checkpoints、loss curve、TensorBoard log）。

---

### `scripts/`
- 執行腳本（作為專案的 entry points）。
- `train.py`: 主訓練腳本，負責載入 config、初始化 trainer 並執行訓練流程。
- 可再新增 `predict.py`, `test.py` 等作為其他任務入口。

---

### `utils/`
- 工具類函式、共用邏輯封裝。
- `logger_util.py`: 日誌紀錄與格式化功能（如初始化 logger、記錄訓練資訊）。
- 其他建議可新增：
  - `model_util.py`: 包含 `prepare_model`, `load_checkpoint`, `save_model` 等功能。
  - `metrics.py`: 評估指標（如 accuracy、IoU、dice_score 等）。

---

### `tests/`
- Unit test
- example: test_logger.py

---

## ✅ 開發建議
- 保持核心邏輯 (`core/`) 與執行腳本 (`scripts/`) 的分離。
- 將重複使用的功能模組化（放進 `utils/`）。

---
