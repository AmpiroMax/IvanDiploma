# Ivan — интерпретация ЭРТ в геологические разрезы

Нейросетевая модель для предсказания геологических разрезов по данным электроразведки (ERT). Архитектура: Perceiver-IO (set encoder → grid decoder).

---

## Установка

### Локально (Windows / Linux)

```bash
# Клонировать репозиторий
git clone <repo_url>
cd IVAN

# Создать виртуальное окружение (опционально)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux

# Установить зависимости
pip install -r requirenments.txt

# Установить пакет в режиме разработки
pip install -e .
```

### Google Colab

1. Загрузите репозиторий в Colab (через Git или ZIP).
2. В первой ячейке выполните:

```python
!pip install -r requirenments.txt
!pip install -e .
```

3. Подключите Google Drive (если данные на Drive):

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Положите данные в `IVAN/data/processed/` (см. раздел «Структура данных»).

---

## Структура проекта

| Путь | Назначение |
|------|------------|
| `iternet/` | Основной пакет |
| `iternet/io/` | Чтение входных/выходных файлов |
| `iternet/train.py` | Обучение и валидация |
| `iternet/scripts/train_batch.py` | Скрипт batch-обучения |
| `iternet/export_ie2.py` | Экспорт предсказаний в .ie2 |
| `notebooks/maga_pipe.ipynb` | Ноутбук для запуска пайплайна |
| `iternet/config.py` | Конфиги экспериментов |
| `data/processed/` | Обучающие и тестовые данные |

---

## Скрипты чтения и записи (`iternet/io/`)

В папке два парсера — оба **актуальны** для текущего формата данных.

### `ie2d.py` — входные данные ЭРТ (`.dat`)

**Назначение:** чтение результатов электроразведки.

| Что парсит | Описание |
|------------|----------|
| Заголовок | sys_path, electrode_spacing, Type of measurement (0=ρa, 1=R), число измерений |
| Строки | `3 xa za xm zm xn zn value` или `4 xa za xb zb xm zm xn zn value` |

**Текущий формат данных** (`data/processed/.../*.dat`):
- 3-электродная схема: `3 xa za xm zm xn zn value` (B на бесконечности)
- `value` — сопротивление или ρa (по заголовку)
- Функция: `parse_ie2d_res(path)` → `IE2DResData`

**Куда смотреть:** `ie2d.py` — если меняется формат `.dat` или добавляются новые поля.

---

### `ie2.py` — геологические модели (`.ie2`)

**Назначение:** чтение целевых разрезов (полигоны, Rho, цвета).

| Что парсит | Описание |
|------------|----------|
| Заголовок | title, nbodies, npoints (строка 3 или с "Nbodies,Npoints") |
| Тела | Rho, Eta, Npoints, color, hatch + индексы точек |
| Точки | `x z - idx` (новый формат) или `x z - idx - th point X,Z` (старый) |

**Текущий формат данных** (`data/processed/.../models/*.ie2`):
- Заголовок на строке 3: `3 34 0 1.5 -1`
- Точки: `x z - idx` (без "th point")
- Функция: `parse_ie2_model(path)` → `IE2Model`

**Куда смотреть:** `ie2.py` — если меняется формат `.ie2` или структура точек.

---

### Кратко

| Файл | Формат | Роль в пайплайне |
|------|--------|------------------|
| `ie2d.py` | `.dat` | Вход модели (измерения ЭРТ) |
| `ie2.py` | `.ie2` | Целевая разметка (ground truth) |

### Выходные файлы

- **`iternet/export_ie2.py`** — экспорт предсказаний в `.ie2`:
  - `export_prediction_to_ie2(...)` — маска → полигоны → `.ie2`
  - `ExportConfig`: `min_area_cells`, `simplify_step`

---

## Обучение и валидация

- **`iternet/train.py`** — функция `train_segmentation()`:
  - Обучение по батчам
  - Валидация после каждой эпохи
  - Loss: CE + Dice + Boundary (Kervadec)
  - Логи в TensorBoard, картинки валидации

- **`iternet/scripts/train_batch.py`** — CLI для batch-обучения:
  - Поиск пар (dat, ie2) в `data/processed/train` и `data/processed/test`
  - Сохранение модели в `iternet/runs/`

---

## Скрипты запуска

### Windows

```cmd
scripts\train.bat
scripts\train.bat 20 8
```

### Linux / macOS

```bash
chmod +x scripts/train.sh
./scripts/train.sh
./scripts/train.sh 20 8
```

### Прямой вызов Python

```bash
python -m iternet.scripts.train_batch --data_dir data/processed --epochs 50 --batch_size 4 --device cuda
```

### Ноутбук

```bash
jupyter notebook notebooks/maga_pipe.ipynb
```

Или через `scripts/run_notebook.bat` (Windows).

---

## Конфиги (`iternet/config.py`)

### DataConfig

| Параметр | Описание |
|----------|----------|
| `ie2d_res_path` | Путь к `.dat` (ЭРТ) |
| `ie2_model_path` | Путь к `.ie2` (разрез) |
| `value_kind` | `"auto"` / `"voltage"` / `"resistance"` / `"rho_a"` |
| `current_a` | Ток (А) для пересчёта напряжения в ρa |

### GridConfig

| Параметр | Описание |
|----------|----------|
| `look_nx`, `look_nz` | Разрешение сетки (256×128) |
| `x_min`, `x_max`, `z_min`, `z_max` | Границы области (м) |

### ModelConfig

| Параметр | Описание |
|----------|----------|
| `token_dim`, `latent_dim` | Размерности энкодера |
| `num_latents` | Число латентных векторов |
| `num_layers`, `num_heads` | Слои и головы attention |
| `dropout` | Dropout |

### TrainConfig

| Параметр | Описание |
|----------|----------|
| `batch_size`, `epochs`, `lr`, `weight_decay` | Параметры обучения |
| `device` | `"cuda"` / `"cpu"` |
| `log_dir` | Папка для TensorBoard |
| `ignore_index` | Класс фона (обычно 0) |
| `boundary_weight_factor` | Множитель loss у границ (3.0) |
| `boundary_weight_radius` | Радиус окрестности границы (пиксели) |
| `ce_weight`, `dice_weight`, `boundary_loss_weight` | Веса CE, Dice, Boundary Loss |
| `log_every_steps` | Частота логирования |

---

## Структура данных

```
data/processed/
├── train/
│   ├── electrical_resistivity_tomography/   # .dat (224.dat, 001.dat, ...)
│   └── models/                              # .ie2 (224.ie2, 001.ie2, ...)
└── test/
    ├── electrical_resistivity_tomography/
    └── models/
```

Пары подбираются по имени файла (например, `224.dat` ↔ `224.ie2`).

---

## Google Colab: пошагово

### 1. Загрузка репозитория

```python
# Вариант A: из Git
!git clone https://github.com/.../IVAN.git
%cd IVAN

# Вариант B: из ZIP на Drive
# Распакуйте архив в /content/IVAN
%cd /content/IVAN
```

### 2. Установка

```python
!pip install -r requirenments.txt
!pip install -e .
```

### 3. Данные

**Вариант A — с Google Drive:**

```python
from google.colab import drive
drive.mount('/content/drive')

# Скопировать данные в проект
!cp -r /content/drive/MyDrive/IVAN_data/processed /content/IVAN/data/
```

**Вариант B — загрузка ZIP:**

```python
from google.colab import files
uploaded = files.upload()  # Выберите data_processed.zip
!unzip data_processed.zip -d data/
```

### 4. Обучение

```python
!python -m iternet.scripts.train_batch --data_dir data/processed --epochs 50 --batch_size 4 --device cuda
```

### 5. TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir iternet/runs
```

### 6. Сохранение результатов

```python
# Скачать модель и логи
from google.colab import files
!zip -r results.zip iternet/runs data/outputs
files.download('results.zip')
```

---

## Выходы обучения

- **`iternet/runs/YYYY-MM-DD_HH-MM-SS/`** — логи TensorBoard
- **`iternet/runs/.../val_images/epoch_0000/`, `epoch_0001/`** — картинки валидации
- **`iternet/runs/model.pt`** — веса модели (после `train_batch`)
- **`data/outputs/`** — экспорт в `.ie2` (из ноутбука или скриптов)

---

## Citation

Дипломный проект.
