# FlowMetric Temporal

`FlowMetric Temporal` — отдельный обучаемый модуль для классификации коротких
эпизодов взаимодействия сотрудника, клиента и стола. Он дополняет основной
FlowMetric: основной проект находит людей, назначает им роли и строит
траектории, а этот проект анализирует динамику уже сформированного эпизода.

Модель относит каждый эпизод к одному из четырёх классов:

| Класс | Смысл |
|---|---|
| `service` | сотрудник действительно обслуживает выбранного клиента |
| `passing` | сотрудник проходит рядом, не начиная обслуживание |
| `waiting` | сотрудник или клиент остаётся рядом без обслуживания |
| `cleanup` | сотрудник убирает или подготавливает стол |

Это специализированная video/trajectory-модель, а не VLM. На production-инференсе
она не отправляет изображения во внешний API и не использует текстовые промпты.
VLM можно подключить отдельно как инструмент предварительной разметки, поиска
сложных негативных примеров или teacher-модель.

## Зачем нужен отдельный temporal-классификатор

Один кадр часто не позволяет отличить обслуживание от похожих ситуаций:

- сотрудник просто стоит рядом со столом;
- проходит между столами;
- обслуживает соседнего клиента;
- разговаривает с коллегой;
- остаётся рядом после завершения обслуживания;
- забирает посуду без взаимодействия с клиентом.

Различие проявляется во времени: меняются расстояния, скорости, положение людей
относительно стола и визуальное содержимое crops. Поэтому модель одновременно
получает изображения и геометрию траекторий на последовательности кадров.

## Текущее состояние

Реализован первый полностью исполняемый baseline:

1. импорт ручной JSONL-разметки или bootstrap из текущего benchmark FlowMetric;
2. синхронное извлечение `staff`, `customer` и `table` crops;
3. построение 20 траекторных признаков;
4. обучение CNN + temporal Transformer;
5. выбор checkpoint по validation macro-F1;
6. пакетный инференс с вероятностями всех классов;
7. строгая проверка схемы датасета и версии checkpoint;
8. end-to-end команда для текущего видео FlowMetric.

Baseline предназначен для проверки полного контура и начала сбора датасета.
Он пока не является готовой промышленной моделью: текущий benchmark содержит
только один короткий ролик и две наблюдаемые GT-пары обслуживания.

## Архитектура решения

```text
                  FlowMetric tracks.json
                           +
                   размеченный интервал
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     staff crops      customer crops    table crops
          │                │                │
          └──── shared CNN encoder ─────────┘
                           │
              3 × visual embeddings
                           │
tracks ──► 20 trajectory features ──► trajectory MLP
                           │
                  feature concatenation
                           │
          learned positional embeddings
                           │
              temporal Transformer encoder
                           │
                 masked temporal pooling
                           │
              service / passing / waiting / cleanup
```

Визуальный encoder общий для всех трёх потоков. Это уменьшает число параметров
и заставляет модель использовать сопоставимое пространство признаков для людей
и стола. Каждый временной шаг содержит:

```text
staff_embedding + customer_embedding + table_embedding + trajectory_embedding
```

Конфигурация по умолчанию:

| Параметр | Значение |
|---|---:|
| размер одного visual embedding | 96 |
| размер trajectory embedding | 32 |
| общий hidden size | 320 |
| attention heads | 8 |
| Transformer layers | 3 |
| dropout | 0.15 |
| максимальное число кадров | 64 |
| число классов | 4 |

Условие конфигурации: `hidden_dim = 3 × visual_dim + trajectory_dim`, а
`hidden_dim` должен делиться на число attention heads.

## Связь с основным FlowMetric

FlowMetric Temporal не запускает детектор и трекер самостоятельно. Он ожидает:

- исходное видео;
- `tracks.json`, созданный FlowMetric;
- идентификаторы сотрудника и клиента;
- границы эпизода в кадрах;
- необязательный `table_box`.

Ответственность модулей разделена так:

| FlowMetric | FlowMetric Temporal |
|---|---|
| детекция людей | классификация готового эпизода |
| tracking и ReID | временное моделирование взаимодействия |
| назначение ролей | различение service и hard negatives |
| генерация кандидатов | вероятность каждого класса |
| карта столов | использование table crop и геометрии |

Temporal-модель не должна создавать новые ID или изменять границы кандидата.
В будущей интеграции она будет verifier-компонентом основного FlowMetric.

## Требования и установка

- Python 3.10 или новее;
- NumPy;
- OpenCV без GUI;
- SciPy;
- PyTorch 2.2 или новее.

```bash
cd /Users/alkselsv/Projects/FlowMetricTemporal

python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

python -m unittest discover -s tests -v
flowmetric-temporal --help
```

`--device auto` выбирает CUDA, затем Apple MPS, затем CPU. Можно потребовать
конкретное устройство через `--device cpu`, `--device mps`, `--device cuda` или
`--device cuda:N`. Если явно указанное устройство недоступно, команда завершится
ошибкой вместо незаметного перехода на CPU.

## Быстрый end-to-end запуск на текущих данных

Текущий репозиторий FlowMetric уже содержит исходное видео, benchmark GT,
`tracks.json` и карту столов. Полный демонстрационный прогон запускается одной
командой:

```bash
cd /Users/alkselsv/Projects/FlowMetricTemporal

flowmetric-temporal current-data \
  --flowmetric-root /Users/alkselsv/Projects/FlowMetric \
  --out runs/current-data \
  --device auto \
  --epochs 30
```

Команда последовательно выполняет:

```text
anchor mapping
    → bootstrap annotations
    → crop/trajectory extraction
    → training
    → best checkpoint
    → validation predictions
```

Результат:

```text
runs/current-data/
  annotations.jsonl
  episodes/
    manifest.jsonl
    gt_service_*.npz
    weak_passing_*.npz
    weak_waiting_*.npz
    weak_cleanup_*.npz
  best.pt
  val_predictions.jsonl
  run.json
```

Назначение файлов:

| Файл | Содержимое |
|---|---|
| `annotations.jsonl` | исходные GT и weak-label эпизоды с аудитом происхождения |
| `episodes/*.npz` | crops, траектории, mask и номера кадров |
| `episodes/manifest.jsonl` | индекс извлечённых эпизодов для Dataset/DataLoader |
| `best.pt` | checkpoint лучшей эпохи по validation macro-F1 |
| `val_predictions.jsonl` | вероятности классов для validation split |
| `run.json` | пути, количество эпизодов и итоговые метрики запуска |

Для быстрой проверки контура можно уменьшить объём вычислений:

```bash
flowmetric-temporal current-data \
  --flowmetric-root /Users/alkselsv/Projects/FlowMetric \
  --out runs/smoke \
  --frames 4 \
  --crop-size 32 \
  --epochs 1 \
  --batch-size 4 \
  --device cpu
```

Такой запуск проверяет код, но его checkpoint не следует использовать как
обученную модель.

## Как работает bootstrap текущих данных

Команда `bootstrap-current` создаёт проверяемую стартовую разметку:

```bash
flowmetric-temporal bootstrap-current \
  --flowmetric-root /Users/alkselsv/Projects/FlowMetric \
  --out runs/current-data/annotations.jsonl
```

По умолчанию используются:

```text
benchmark/ground_truth.json
config/tables.json
data/hf_*.mp4
data/output/service_verifier_tables_run/tracks.json
```

Другой результат трекинга можно передать путём относительно корня FlowMetric:

```bash
flowmetric-temporal bootstrap-current \
  --flowmetric-root /Users/alkselsv/Projects/FlowMetric \
  --tracks data/output/local/tracks.json \
  --out runs/local-tracks/annotations.jsonl
```

Bootstrap состоит из следующих этапов.

### 1. Сопоставление GT-ID и track-ID

GT содержит точки головы на опорных кадрах, а `tracks.json` — боксы с ID
трекера. Сопоставление повторяет benchmark-логику FlowMetric:

- оставляет боксы, содержащие точку головы с небольшим допуском;
- штрафует короткие фрагменты траекторий;
- исключает вложенные короткие боксы;
- решает глобальное назначение алгоритмом Hungarian;
- запрещает повторное использование одного track-ID.

### 2. Положительные эпизоды

`service_start` и `service_end` объединяются по паре `staff/customer`.
Интервалы переводятся из секунд в номера кадров. Нулевой левосторонне
цензурированный эпизод, заканчивающийся на кадре 0, пропускается, поскольку в
видео нет наблюдаемой динамики обслуживания.

На текущем benchmark получаются две положительные пары `service`.

### 3. Weak hard negatives

Для пар сотрудник–клиент строятся скользящие окна и вычисляются:

- среднее и минимальное расстояние между людьми в высотах тела;
- скорость движения сотрудника;
- доля кадров, в которых сотрудник находится около выбранного стола.

По этим признакам ранжируются кандидаты `passing`, `waiting` и `cleanup`.
По умолчанию выбирается до шести кандидатов каждого класса.

Каждая такая строка содержит:

```json
{
  "label_source": "weak_geometry_heuristic",
  "review_required": true,
  "weak_label_scores": {
    "passing": 1.42,
    "waiting": 0.71,
    "cleanup": 1.08
  },
  "window_stats": {
    "mean_distance": 1.28,
    "min_distance": 0.14,
    "staff_speed": 1.78,
    "near_table_ratio": 0.5
  }
}
```

Важно: weak labels — это очередь кандидатов для ручной проверки, а не ground
truth. Особенно ненадёжен класс `cleanup`: геометрия показывает присутствие
сотрудника возле стола и удалённость выбранного клиента, но не доказывает факт
уборки. До ручной проверки эти примеры годятся только для проверки pipeline и
экспериментов с weak supervision.

### 4. Ограничение демонстрационной валидации

Текущий ролик длится примерно 15 секунд. Bootstrap вынужден формировать `train`
и `val` из одного видео. Такие выборки визуально и временно зависимы, поэтому
полученный validation macro-F1 не измеряет обобщение модели. Для настоящей
оценки split необходимо делать по исходным видео, камерам или заведениям.

## Подготовка собственного датасета

Для production-экспериментов рекомендуется создавать JSONL вручную или через
отдельный интерфейс разметки. Одна строка описывает одну пару и один интервал:

```json
{
  "episode_id": "venue01_clip001_000120_000240",
  "label": "service",
  "split": "train",
  "video": "/data/venue01/clip001.mp4",
  "tracks": "/data/venue01/tracks.json",
  "start_frame": 120,
  "end_frame": 240,
  "staff_id": "7",
  "customer_id": "12",
  "table_box": [410, 280, 790, 610]
}
```

Поля:

| Поле | Обязательное | Описание |
|---|---|---|
| `episode_id` | да | уникальный filename-safe ID без `/` и `\\` |
| `label` | да | `service`, `passing`, `waiting` или `cleanup` |
| `split` | нет | `train` по умолчанию; для обучения нужен также `val` |
| `video` | да | путь к исходному видео |
| `tracks` | да | путь к совместимому `tracks.json` |
| `start_frame` | да | первый кадр интервала, включительно |
| `end_frame` | да | последний кадр интервала, включительно |
| `staff_id` | да | ID сотрудника внутри `tracks.json` |
| `customer_id` | да | ID клиента внутри `tracks.json` |
| `table_box` | нет | `[x1, y1, x2, y2]`; без него table crops заполняются нулями |

Дополнительные поля разрешены в аннотациях и могут использоваться для аудита,
но extractor переносит в manifest только `episode_id`, `label`, `split` и путь
к `.npz`. Поэтому исходный `annotations.jsonl` следует сохранять вместе с
датасетом.

Пути разрешаются относительно текущего рабочего каталога процесса, а не
автоматически относительно расположения файла аннотаций. Для воспроизводимых
запусков удобнее использовать абсолютные пути или запускать команды из одного
и того же каталога.

## Требования к `tracks.json`

Ожидается стандартный результат FlowMetric:

```json
{
  "fps": 24.0,
  "width": 1920,
  "height": 1080,
  "n_frames": 361,
  "tracks": {
    "7": {
      "role": "staff",
      "frames": [120, 121, 122],
      "boxes": [
        [100.0, 200.0, 220.0, 520.0],
        [102.0, 201.0, 222.0, 521.0],
        [105.0, 202.0, 225.0, 522.0]
      ]
    }
  }
}
```

Для каждого ID массивы `frames` и `boxes` должны иметь одинаковую длину,
кадры должны быть строго возрастающими, а координаты — конечными числами.
Extractor линейно интерполирует боксы на выбранные временные точки и помечает
точку невидимой, если ближайшее наблюдение дальше половины секунды.

## Извлечение episode tensors

```bash
flowmetric-temporal extract \
  --annotations annotations.jsonl \
  --out data/episodes \
  --frames 16 \
  --crop-size 112
```

Из каждого интервала равномерно выбирается фиксированное количество кадров.
Для каждого временного шага:

1. интерполируются боксы сотрудника и клиента;
2. боксы расширяются на 15% и ограничиваются границами кадра;
3. crops переводятся из BGR в RGB;
4. каждый crop масштабируется в квадрат `crop-size × crop-size`;
5. строятся траекторные признаки;
6. создаётся boolean mask совместной видимости.

Если `table_box` отсутствует, table crop состоит из нулей. Если человек не
считается видимым на выбранном шаге, соответствующий crop также заполняется
нулями, а временной шаг исключается mask при temporal pooling и attention.

Каждый `.npz` содержит:

| Массив | Тип | Форма | Смысл |
|---|---|---|---|
| `staff` | `uint8` | `[T,H,W,3]` | RGB crops сотрудника |
| `customer` | `uint8` | `[T,H,W,3]` | RGB crops клиента |
| `table` | `uint8` | `[T,H,W,3]` | RGB crops стола |
| `trajectory` | `float32` | `[T,20]` | траекторные признаки |
| `mask` | `bool` | `[T]` | совместная видимость людей |
| `frames` | integer | `[T]` | номера выбранных кадров |

При загрузке изображения преобразуются в `[T,3,H,W]`, `float32` и диапазон
`[0,1]`. Dataset отклоняет несовместимые dtype, shape и эпизоды без единого
валидного временного шага.

## Траекторные признаки

Координаты и размеры нормализуются на ширину и высоту кадра.

| Индексы | Признаки |
|---|---|
| 0–3 | `staff_cx`, `staff_cy`, `staff_w`, `staff_h` |
| 4–7 | `customer_cx`, `customer_cy`, `customer_w`, `customer_h` |
| 8–11 | `table_cx`, `table_cy`, `table_w`, `table_h` |
| 12–13 | направление от сотрудника к клиенту: `dx`, `dy` |
| 14 | евклидово расстояние сотрудник–клиент |
| 15–16 | скорость центра сотрудника: `vx`, `vy` |
| 17–18 | скорость центра клиента: `vx`, `vy` |
| 19 | `joint_visible` |

Скорости вычисляются между выбранными временными точками в нормализованных
координатах за секунду. Если стол не задан, его четыре признака равны нулю.

## Обучение

```bash
flowmetric-temporal train \
  --manifest data/episodes/manifest.jsonl \
  --out checkpoints/best.pt \
  --epochs 30 \
  --batch-size 16 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --device auto \
  --workers 0 \
  --seed 17
```

Текущий training loop использует:

- inverse-frequency class weights;
- `CrossEntropyLoss`;
- `AdamW`;
- cosine learning-rate schedule;
- gradient clipping с нормой 1.0;
- лучший checkpoint по validation macro-F1;
- детерминированные начальные seed для Python, NumPy и PyTorch.

После каждой эпохи печатается JSON:

```json
{"epoch": 3, "accuracy": 0.75, "macro_f1": 0.68, "train_loss": 0.91}
```

Macro-F1 вычисляется по всем четырём классам. Если класс отсутствует в
validation split, его F1 равен нулю; это помогает обнаруживать неполную
валидационную выборку, но делает итоговую метрику низкой.

### Что хранится в checkpoint

`best.pt` содержит:

- `format_version`;
- фиксированный порядок labels;
- фиксированный порядок trajectory features;
- конфигурацию модели;
- `state_dict`;
- номер лучшей эпохи;
- validation-метрики лучшей эпохи.

Runtime отказывается загружать checkpoint с неизвестной версией, другим
порядком классов или несовместимым набором признаков. Это защищает от тихой
ошибки, когда новый код интерпретирует старые веса по другой схеме.

## Инференс

```bash
flowmetric-temporal predict \
  --manifest data/episodes/manifest.jsonl \
  --checkpoint checkpoints/best.pt \
  --out predictions.jsonl \
  --split val \
  --batch-size 16 \
  --device auto
```

Без `--split` обрабатываются все строки manifest. Формат результата:

```json
{
  "episode_id": "venue01_clip001_000120_000240",
  "label": "service",
  "confidence": 0.73,
  "probabilities": {
    "service": 0.73,
    "passing": 0.11,
    "waiting": 0.10,
    "cleanup": 0.06
  }
}
```

`confidence` — максимальная softmax-вероятность, а не откалиброванная
вероятность правильности. Для production нужно отдельно выполнить temperature
scaling или другую калибровку и определить порог `abstain`.

## CLI

### `extract`

| Аргумент | По умолчанию | Назначение |
|---|---:|---|
| `--annotations` | обязательный | исходный JSONL |
| `--out` | обязательный | каталог `.npz` и manifest |
| `--frames` | 16 | число временных шагов |
| `--crop-size` | 112 | сторона квадратного crop |

### `train`

| Аргумент | По умолчанию | Назначение |
|---|---:|---|
| `--manifest` | обязательный | manifest извлечённого датасета |
| `--out` | обязательный | путь к checkpoint |
| `--epochs` | 30 | число эпох |
| `--batch-size` | 16 | размер batch |
| `--learning-rate` | `3e-4` | начальный learning rate |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--device` | `auto` | `auto`, `cpu`, `mps`, `cuda`, `cuda:N` |
| `--workers` | 0 | DataLoader workers |
| `--seed` | 17 | seed запуска |

### `predict`

| Аргумент | По умолчанию | Назначение |
|---|---:|---|
| `--manifest` | обязательный | manifest эпизодов |
| `--checkpoint` | обязательный | обученный checkpoint |
| `--out` | обязательный | JSONL результата |
| `--split` | все строки | фильтр split |
| `--batch-size` | 16 | размер batch |
| `--device` | `auto` | вычислительное устройство |

### `bootstrap-current`

| Аргумент | По умолчанию | Назначение |
|---|---:|---|
| `--flowmetric-root` | обязательный | корень текущего FlowMetric |
| `--out` | обязательный | выходной annotations JSONL |
| `--tracks` | `data/output/service_verifier_tables_run/tracks.json` | tracks относительно корня |
| `--max-weak-per-class` | 6 | максимум кандидатов каждого negative-класса |

### `current-data`

| Аргумент | По умолчанию | Назначение |
|---|---:|---|
| `--flowmetric-root` | обязательный | корень FlowMetric |
| `--out` | обязательный | каталог всего запуска |
| `--tracks` | verifier tables run | tracks относительно корня |
| `--frames` | 16 | временных шагов на эпизод |
| `--crop-size` | 112 | размер crop |
| `--epochs` | 30 | эпох обучения |
| `--batch-size` | 8 | размер batch |
| `--device` | `auto` | вычислительное устройство |
| `--prepare-only` | false | остановиться после построения tensors |

## Рекомендации по разметке

### Правило выбора пары

Один эпизод всегда относится к конкретной паре `staff_id/customer_id`. Если
сотрудник обслуживает соседнего клиента, то для неправильной пары это hard
negative, а для правильной пары — `service`. Оба примера полезно сохранить.

### Hard negatives

| Сцена | Рекомендуемая метка |
|---|---|
| сотрудник проходит между столами | `passing` |
| сотрудник кратко пересекает область клиента | `passing` |
| сотрудник стоит рядом без взаимодействия | `waiting` |
| сотрудник остаётся после обслуживания | следующий эпизод `waiting` или `cleanup` по фактическому действию |
| уборка посуды или поверхности | `cleanup` |
| подготовка пустого стола | `cleanup` |
| обслуживание соседнего клиента | `waiting` для неправильной пары и `service` для правильной |

Разговор с коллегой без выбранного клиента не соответствует текущему контракту
парного эпизода. Его следует отфильтровывать генератором кандидатов. Если такие
сцены должны классифицироваться моделью, нужно осознанно добавить класс `other`,
обновить schema/checkpoint version и переразметить датасет.

### Границы эпизодов

- захватывайте небольшой контекст до и после действия;
- используйте одинаковое правило границ для всех классов;
- не объединяйте несколько разных действий в один интервал;
- длинное присутствие рядом лучше делить на однородные короткие эпизоды;
- отмечайте цензурированные границы отдельно в исходной системе разметки.

### Train/validation/test

Нельзя случайно делить соседние окна одного видео между split: они содержат
почти одинаковые кадры и создают сильную утечку. Предпочтительный порядок:

1. test — полностью отдельные заведения или камеры;
2. validation — отдельные исходные ролики;
3. train — оставшиеся ролики;
4. соседние окна одного ролика всегда находятся в одном split.

Дополнительно следует контролировать баланс камер, освещения, плотности зала,
типов формы сотрудников, ракурсов столов и длительности эпизодов.

## Оценка качества

Основная offline-метрика baseline — macro-F1, поскольку классы несбалансированы
и ошибки по редким hard negatives важны не меньше общей accuracy.

Для реальной оценки рекомендуется считать:

- per-class precision, recall и F1;
- confusion matrix;
- service false-positive rate на каждом типе hard negatives;
- service recall по дальности, окклюзии и длительности;
- метрики отдельно по камерам и заведениям;
- calibration error и coverage при использовании abstain threshold;
- end-to-end event precision/recall после интеграции с FlowMetric.

Нужно различать две оценки:

1. **classifier-only** — эпизод и правильные ID уже даны;
2. **end-to-end** — учитываются ошибки детектора, tracking, ролей и генератора
   кандидатов.

Высокая classifier-only метрика не гарантирует качество всей системы.

## Структура репозитория

```text
FlowMetricTemporal/
  pyproject.toml
  README.md
  examples/
    annotations.jsonl
  src/flowmetric_temporal/
    schema.py          labels и порядок trajectory features
    features.py        интерполяция боксов и геометрические признаки
    extract.py         видео → crops/trajectory tensors
    dataset.py         строгая загрузка `.npz`
    model.py           shared CNN + temporal Transformer
    train.py           обучение и метрики
    runtime.py         устройства и checkpoint contract
    infer.py           пакетный инференс
    bootstrap.py       адаптер текущего benchmark FlowMetric
    end_to_end.py      единый bootstrap→train→predict workflow
    cli.py             команды `flowmetric-temporal`
  tests/
    test_bootstrap.py
    test_dataset.py
    test_features.py
    test_model.py
    test_train.py
```

## Тестирование

```bash
python -m unittest discover -s tests -v
```

Тесты проверяют:

- anchor mapping GT-ID → track-ID;
- объединение `service_start/service_end`;
- пропуск нулевого цензурированного интервала;
- интерполяцию боксов и visibility mask;
- точный контракт 20 траекторных признаков;
- загрузку `.npz` и преобразование crops;
- forward/backward модели;
- расчёт accuracy и macro-F1.

Отдельно полезен быстрый интеграционный прогон `current-data` с одним epoch,
приведённый выше.

## Известные ограничения

- visual encoder обучается с нуля, pretrained backbone пока отсутствует;
- нет image/video аугментаций;
- нет отдельного класса `other`;
- нет калибровки confidence и режима abstain;
- нет UI для просмотра и исправления weak labels;
- table crop статичен внутри эпизода;
- bootstrap использует геометрию, но не подтверждает фактическую уборку;
- текущая выборка слишком мала для измерения обобщения;
- основной FlowMetric пока не вызывает checkpoint как `ServiceVerifier`;
- модель классифицирует предложенный эпизод, но не локализует его границы.

## Типичные проблемы

### `manifest has no rows for split 'val'`

В annotations отсутствуют строки `"split": "val"`. Добавьте независимый
validation split; не копируйте туда те же эпизоды из train.

### `mask must select at least one time step`

В выбранном интервале сотрудник и клиент не наблюдаются одновременно. Проверьте
ID, границы эпизода и качество tracking.

### `checkpoint labels do not match runtime labels`

Checkpoint создан с другой схемой классов. Используйте совместимую версию кода
или переобучите модель после изменения schema.

### Запрошенное устройство недоступно

Используйте `--device auto` или установите PyTorch со сборкой для нужного
ускорителя. Явный `mps`/`cuda` специально не переключается на CPU автоматически.

### Очень высокая validation-метрика на одном видео

Скорее всего, соседние временные окна попали одновременно в train и validation.
Переделайте split по исходным видео или камерам.

## План развития

Ближайшие практические этапы:

1. интерфейс просмотра и ручной коррекции bootstrap-кандидатов;
2. сбор нескольких независимых видео и camera-level split;
3. pretrained visual encoder и аугментации;
4. sampler с балансом классов, камер и типов hard negatives;
5. temperature scaling и abstain threshold;
6. адаптер checkpoint к интерфейсу `ServiceVerifier` основного FlowMetric;
7. end-to-end benchmark с учётом candidate recall;
8. distillation от video/VLM teacher при наличии достаточной разметки.

Главный следующий шаг — не усложнение архитектуры, а получение проверенной
эпизодной разметки. На небольшом и зависимом датасете более крупная модель лишь
быстрее запомнит конкретную камеру и людей.
