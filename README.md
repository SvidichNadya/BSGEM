# BSGEM-HFE: Generative Bayesian Model of Information-Emotional Dynamics

This repository contains the implementation of the **BSGEM-HFE** (Bayesian Structural Generative Econometric Model with Human Factor Extraction) proposed in the research paper *"Generative Bayesian Model of Information-Emotional Dynamics"*.

The model integrates:
- Econophysics of information fields (potential, gradient, quadratic energy, cognitive memory)
- FJ opinion dynamics with self-deception
- Social Suggestibility Index (SSI) and Narrative Adoption Index (NAI)
- Human Factor Extraction (HFE) from residuals
- Endogenous narrative self-reproduction
- Variational autoencoder (VAE) with ELBO
- Dynamic causal graphs and sliding-window HMM
- Time series cross-validation

It is designed for forecasting irrational behaviour in financial markets based on textual data (trader diaries, news, social media).

---

## ⚙️ Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/SvidichNadya/BSGEM
   cd BSGEM
   ```

2. **Install Python dependencies**  
   All required packages will be automatically installed when you run the script for the first time.  
   Alternatively, manually install:
   ```bash
   pip install torch transformers sentence-transformers bertopic scikit-learn lightgbm shap pykalman nltk pandas numpy umap-learn hdbscan matplotlib seaborn tqdm joblib scipy statsmodels econml hmmlearn pytorch-forecasting
   ```

3. **Prepare your dataset**  
   The script expects a CSV file with at least three columns:  
   - `timestamp` (datetime)  
   - `text` (string, the message content)  
   - `label` (numeric, target variable, e.g., 0/1 for irrationality)

   Example:
   ```csv
   timestamp,text,label
   2026-06-01 09:30:15,"Утром всё было чётко по плану...",1
   ```

4. **Run the model**  
   ```bash
   python BSGEM_model.py dataset.csv
   ```

   The script will:
   - Extract NLP features (sentiment, anxiety, narrative topics, etc.)
   - Apply Kalman filtering and sliding-window HMM (no future leakage)
   - Perform 3-fold time series cross-validation
   - Train baseline models (LinearRegression, RandomForest, LightGBM, VAR(2), Transformer, DeepAR) and the full BSGEM‑HFE model
   - Output metrics (MSE, MAE, R²) and CATE estimates
   - Show plots (actual vs predicted, training curves, CATE distribution)

5. **Output**  
   - `report.json` – summary metrics for all models
   - Interactive plots displayed during execution

---

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

All dependencies are listed in the import section of the script and will be installed automatically if missing.

---

## Contributing

This is a research prototype. Feel free to open issues or submit pull requests via GitHub.  
For major changes, please discuss first.


## 🔗 Citation

If you use this code in your research, please cite the original.

---

# BSGEM-HFE: Генеративная байесовская модель информационно-эмоциональной динамики

## Описание

Репозиторий содержит реализацию модели **BSGEM‑HFE** (байесовская структурная генеративная эконометрическая модель с извлечением человеческого фактора), предложенной в научной работе *«Генеративная байесовская модель информационно-эмоциональной динамики»*.

Модель объединяет:
- эконофизику информационных полей (потенциал, градиент, энергия, когнитивная память),
- динамику мнений Фридкина-Джонсена с самообманом,
- индекс внушаемости общества (SSI) и индекс веры в нарратив (NAI),
- извлечение человеческого фактора из остатков (HFE),
- эндогенное самовоспроизводство нарративов,
- вариационный автокодировщик (VAE) с ELBO,
- динамический причинный граф и скользящее HMM,
- временную кросс-валидацию.

Предназначена для прогнозирования иррационального поведения на финансовых рынках на основе текстовых данных (дневники трейдеров, новости, соцсети).

---

## ⚙️ Установка и запуск

1. **Клонируйте репозиторий**  
   ```bash
   git clone https://github.com/SvidichNadya/BSGEM
   cd BSGEM
   ```

2. **Установите зависимости**  
   При первом запуске необходимые пакеты установятся автоматически.  
   Либо установите вручную:
   ```bash
   pip install torch transformers sentence-transformers bertopic scikit-learn lightgbm shap pykalman nltk pandas numpy umap-learn hdbscan matplotlib seaborn tqdm joblib scipy statsmodels econml hmmlearn pytorch-forecasting
   ```

3. **Подготовьте данные**  
   Скрипт ожидает CSV-файл с как минимум тремя столбцами:  
   - `timestamp` (дата и время)  
   - `text` (текст сообщения)  
   - `label` (числовая целевая переменная, например 0/1 для иррациональности)

   Пример:
   ```csv
   timestamp,text,label
   2026-06-01 09:30:15,"Утром всё было чётко по плану...",1
   ```

4. **Запустите модель**  
   ```bash
   python BSGEM_model.py dataset.csv
   ```

   Скрипт выполнит:
   - извлечение NLP-признаков (сентимент, тревожность, топики нарративов и др.),
   - фильтрацию Калмана и скользящее HMM (без заглядывания в будущее),
   - 3-кратную временную кросс-валидацию,
   - обучение базовых моделей (линейная регрессия, случайный лес, LightGBM, VAR(2), Transformer, DeepAR) и полной BSGEM‑HFE,
   - вывод метрик (MSE, MAE, R²) и оценок CATE,
   - отображение графиков (фактические vs предсказанные, кривые обучения, распределение CATE).

5. **Результаты**  
   - `report.json` – сводные метрики для всех моделей.
   - Интерактивные графики, появляющиеся в процессе выполнения.

---

## Требования

- Python 3.8+
- GPU с поддержкой CUDA (необязательно, для ускорения обучения)

Все зависимости перечислены в секции импорта скрипта и устанавливаются автоматически при необходимости.

---

## Участие в разработке

Это исследовательский прототип. Пожалуйста, открывайте issues или pull request через GitHub.  
Для существенных изменений предварительно обсудите их.

---

## 🔗 Цитирование

При использовании кода в своих исследованиях, пожалуйста, ссылайтесь на оригинал.
