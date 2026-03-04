# Ivan — ERT to 2D matrix regression

Модель для предсказания 2D матрицы по данным электроразведки (ERT).

- **Вход (актуально):** `.npz` с двухканальным полем `matrix_data` размера **`(29, 47, 2)`**
- **Цель:** `.npz` с матрицей `float` размера **`(300, 600)`** (ключ `output_matrix_loaded`)
- **Задача:** регрессия (не сегментация по классам)

---

## Установка

Проект обычно запускается из conda env `ivan`.

```bash
git clone <repo_url>
cd IVAN
# пример (если conda доступна в PATH):
# conda create -n ivan python=3.10 -y
# conda activate ivan
# pip install -r requirenments.txt
# pip install -e .
```

---

## Актуальная структура данных

```text
data/processed/
├── train/
│   ├── input/                               # *.npz (matrix_data: 29x47x2)
│   └── output/                              # *.npz (output_matrix_loaded: 300x600)
└── test/
    ├── input/                               # *.npz
    └── output/                              # *.npz
```

Пары подбираются по имени файла:
`001.npz` ↔ `001.npz` (input ↔ output).

---

## Что внутри `.npz`

### Input

`data/processed/*/input/*.npz`:

- key: `matrix_data`
- dtype: float
- shape: `(29, 47, 2)` (в коде приводится к `(2, 29, 47)` и нормируется в `[0,1]`)

### Output

`data/processed/*/output/*.npz`:

- key: `output_matrix_loaded`
- dtype: float
- shape: `(300, 600)`

---

## Пайплайн данных

1. `iternet/preprocessing.py` загружает `matrix_data` и нормирует вход по каналам в `[0,1]`.
2. Target `output_matrix_loaded` используется **в raw масштабе** (без нормализации).
3. Модель `IternetUNet` предсказывает карты `N/K/D` и проецирует в значение:
   \[
   \hat{y} = K \cdot 10^{N} + D
   \]
   Лосс считается на \(\hat{y}\) в raw-домене.

---

## Конфиги (`iternet/config.py`)

### DataConfig

- `ie2d_res_path`: путь к входному `.npz` (или legacy `.dat`)
- `target_matrix_path`: путь к target `.npz`
- `value_kind`, `current_a`: параметры интерпретации последней колонки измерений

### GridConfig

- `look_nx=600`, `look_nz=300` (по умолчанию)
- `x_min`, `x_max`, `z_min`, `z_max` — физические границы сетки

### ModelConfig

- `in_channels=2`
- `base_channels=32`
- `out_channels=1`

### TrainConfig

- стандартные параметры обучения: `batch_size`, `epochs`, `lr`, `weight_decay`, `device`, `log_dir`
- веса лосса: `mse_weight`, `mae_weight`, `boundary_loss_weight`, `boundary_weight_factor`, `boundary_weight_radius`

---

## Обучение

```bash
python -m iternet.scripts.train_batch --data_dir data/processed --epochs 50 --batch_size 4 --device cuda
```

По умолчанию скрипт тренирует на сетке `600x300`.

Логируемые метрики:

- `loss` (регрессионный loss в raw-домене)
- `MAE`
- `RMSE`

---

## Инференс

Основные функции в `iternet/pipeline.py`:

- `open_training_data(...)`
- `preprocess_data(...)`
- `init_model(..., checkpoint_path=...)`
- `predict_mask(...)` → возвращает **raw float-матрицу** `(300, 600)`

---

## Визуализация

`iternet/viz.py` обновлен для матриц:

- `plot_prediction(...)` — grayscale (`matplotlib`, `cmap="gray"`)
- `plot_target_vs_prediction(...)` — сравнение target/prediction в grayscale

---

## Ноутбук

Актуальный ноутбук: `notebooks/maga_pipe_regression.ipynb`.

