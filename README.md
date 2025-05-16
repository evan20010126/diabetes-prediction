# Data Science Project - Predict Diabetes From Medical Records

Build a predictive model to identify the likelihood of diabetes in patients based on their medical records. By leveraging structured health data such as Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin and so on.

## Slides
1. Proposal & Progress Update: [slides](https://www.canva.com/design/DAGnl981ApY/3E5M8Dbml5FOcDcIu-ktSg/edit?utm_content=DAGnl981ApY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
2. Final Presentation: [slides](https://www.canva.com/design/DAGnlwDe9K4/SGbfeEl9PTIm31ZglRko3g/edit?utm_content=DAGnlwDe9K4&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

> This project is for academic purposes only. Almost all images from the internet are marked its source. If there are any images or data that are not authorized for use, please let me know and I will remove them immediately.

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
- Run PIMA dataset:
1. Run non-data-leakage version 
    ```sh
      sh ./launch/run_pima_non_dataleakage.sh
    ```
2. Run data-leakage version 
    ```sh
      sh ./launch/run_pima_dataleakage.sh
    ```

- Run Frankfurt dataset:`sh ./launch/run_frankfurt.sh`

---

## 📁 Folder and File Descriptions

### `configs/`
- Stores model configurations and training parameters in `.yaml` files.
- **Suggested naming**: `train.yaml`, `test.yaml`, etc.

---

### `cores/`
- Encapsulates the **core logic** of the project (classes and functions for flow control).
- `trainer.py`: Contains the `Trainer` class for training logic (e.g., training loop, validation, saving model).
- `inferencer.py`: Contains the `Inferencer` class for inference logic (e.g., single image prediction, batch inference).

---

### `data/`
- Stores datasets used by the project.
- For example, you can place `diabetes.csv` here.

---

### `launch/`
- Scripts for launching or executing the project.

---

### `models/`
- Defines model architectures.
- Place your PyTorch, TensorFlow, or custom model classes here.
- **Example**: `unet.py`, `resnet_classifier.py`, `tabnet.py`, etc.

---

### `outputs/`
- Stores training or inference results.
- **Suggested subfolders**:
  - `tools/`: Scripts for inference visualization and result analysis.
  - `training/`: Stores training outputs such as checkpoints, loss curves, and TensorBoard logs.

---

### `scripts/`
- Entry-point scripts for execution.
- `train.py`: Main training script responsible for loading configs, initializing the trainer, and running the training process.
- You may also add `predict.py`, `test.py`, etc. for other tasks.

---

### `utils/`
- Utility functions and shared logic.
- `logger_util.py`: Handles logging initialization and formatting (e.g., recording training information).
- **Other suggested modules**:
  - `model_util.py`: Includes functions like `prepare_model`, `load_checkpoint`, and `save_model`.
  - `metrics.py`: Contains evaluation metrics (e.g., accuracy, IoU, Dice score).

---

### `tests/`
- Unit tests for various components.
- **Example**: `test_logger.py`

---

## ✅ Development Suggestions
- Keep core logic in `cores/` and execution logic in `scripts/` separated.
- Modularize reusable functions into `utils/`.
- Organize model definitions in `models/` for better maintainability.

