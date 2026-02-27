# Ivan — ERT to 2D matrix regression

Модель для предсказания 2D матрицы по данным электроразведки (ERT).

- **Вход:** `.dat` (измерения ABMN)
- **Цель:** `.npz` с матрицей `float` размера **`(300, 600)`**
- **Задача:** регрессия (не сегментация по классам)

---

## Установка

```bash
git clone <repo_url>
cd IVAN
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux
pip install -r requirenments.txt
pip install -e .
```

---

## Актуальная структура данных

```text
data/processed/
├── train/
│   ├── electrical_resistivity_tomography/   # *.dat
│   └── models/                              # *.npz (target matrix 300x600)
└── test/
    ├── electrical_resistivity_tomography/   # *.dat
    └── models/                              # *.npz
```

Пары подбираются по имени файла:
`001.dat` ↔ `001.npz`.

---

## Что внутри `.npz`

Ожидается 2D матрица с shape `(300, 600)` и float-значениями.  
В текущих данных, например `data/processed/test/models/001.npz`, хранится массив:

- key: `output_matrix_loaded.npy`
- dtype: `float64`
- shape: `(300, 600)`

---

## Пайплайн данных

1. `iternet/io/ie2d.py` читает `.dat` в `IE2DResData`.
2. `iternet/preprocessing.py`:
   - строит `meas_tokens` из ABMN;
   - загружает target-матрицу из `.npz`;
   - делает нормализацию target для обучения:
     - `signed_log = sign(v) * log1p(abs(v))`
     - затем min-max в `[-1, 1]`.
3. Модель предсказывает нормализованную матрицу.
4. На инференсе выполняется обратное преобразование в исходный масштаб значений.

---

## Конфиги (`iternet/config.py`)

### DataConfig

- `ie2d_res_path`: путь к входному `.dat`
- `target_matrix_path`: путь к target `.npz`
- `value_kind`, `current_a`: параметры интерпретации последней колонки измерений

### GridConfig

- `look_nx=600`, `look_nz=300` (по умолчанию)
- `x_min`, `x_max`, `z_min`, `z_max` — физические границы сетки

### ModelConfig

- `token_dim`, `latent_dim`, `num_latents`, `num_layers`, `num_heads`, `dropout`
- `out_channels=1` (регрессия одной матрицы)

### TrainConfig

- стандартные параметры обучения: `batch_size`, `epochs`, `lr`, `weight_decay`, `device`, `log_dir`

---

## Обучение

```bash
python -m iternet.scripts.train_batch --data_dir data/processed --epochs 50 --batch_size 4 --device cuda
```

По умолчанию скрипт тренирует на сетке `600x300`.

Логируемые метрики:

- `loss` (регрессионный loss в нормализованном пространстве)
- `MAE`
- `RMSE`

---

## Инференс

Основные функции в `iternet/pipeline.py`:

- `open_training_data(...)`
- `preprocess_data(...)`
- `init_model(..., checkpoint_path=...)`
- `predict_mask(...)` → возвращает **денормализованную float-матрицу** `(300, 600)`

---

## Визуализация

`iternet/viz.py` обновлен для матриц:

- `plot_prediction(...)` — grayscale (`matplotlib`, `cmap="gray"`)
- `plot_target_vs_prediction(...)` — сравнение target/prediction в grayscale

---

## Ноутбук

`notebooks/maga_pipe.ipynb` содержит обновленный раздел:

- **Matrix Regression Pipeline (Updated)**
- self-contained пример: путь к `.dat`, `.npz`, чекпоинт, инференс, метрики (MAE/RMSE/MAPE/R2), grayscale-визуализация.

