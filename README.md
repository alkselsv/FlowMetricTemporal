# FlowMetric Temporal

Обучаемый классификатор коротких эпизодов для FlowMetric. Модель получает
последовательности фрагментов сотрудника, клиента и стола вместе с геометрией
траекторий и относит эпизод к одному из классов:

- `service` — обслуживание клиента;
- `passing` — сотрудник проходит рядом;
- `waiting` — сотрудник или клиент находится рядом без обслуживания;
- `cleanup` — уборка или подготовка стола.

Это специализированная video/trajectory-модель, а не VLM. Визуальные признаки
трёх crop-потоков извлекаются общей CNN, объединяются с нормализованными
траекторными признаками и обрабатываются temporal Transformer. VLM можно позднее
подключить как teacher для разметки и hard-negative mining, не добавляя её в
production runtime.

## Состояние

Реализован первый сквозной baseline:

1. `extract` читает видео, стандартный `tracks.json` FlowMetric и JSONL-аннотации;
2. строит синхронные crops и 20 траекторных признаков;
3. `train` обучает модель с class-balanced loss и выбирает checkpoint по macro-F1;
4. `predict` пишет вероятности всех четырёх классов в JSONL;
5. checkpoint и входной формат версионированы и строго проверяются.

Для данных из текущего репозитория FlowMetric также есть bootstrap-конвейер:

- сопоставляет GT-ID с ID из `tracks.json` тем же anchor-алгоритмом, что benchmark;
- превращает `service_start/service_end` в положительные интервалы;
- подбирает стол и генерирует кандидаты `passing/waiting/cleanup`;
- сохраняет происхождение каждой метки и геометрические оценки;
- одной командой выполняет подготовку, обучение и инференс.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

## Формат аннотаций

Одна строка JSONL описывает один эпизод:

```json
{"episode_id":"venue01_clip001_000120_000240","label":"service","split":"train","video":"/data/venue01/clip001.mp4","tracks":"/data/venue01/tracks.json","start_frame":120,"end_frame":240,"staff_id":"7","customer_id":"12","table_box":[410,280,790,610]}
```

`tracks` должен ссылаться на `tracks.json`, который уже создаёт FlowMetric.
`table_box` необязателен; при его отсутствии поток стола заполняется нулями.
Пути в аннотациях могут быть абсолютными или относительными к текущему каталогу.

Разделение `train`/`val` необходимо делать по камерам, заведениям или исходным
видеофайлам, а не случайно по соседним эпизодам. Иначе почти одинаковые кадры
попадут в обе выборки и завысят метрики.

## Запуск

```bash
flowmetric-temporal extract \
  --annotations annotations.jsonl \
  --out data/episodes \
  --frames 16 \
  --crop-size 112

flowmetric-temporal train \
  --manifest data/episodes/manifest.jsonl \
  --out checkpoints/best.pt \
  --device auto

flowmetric-temporal predict \
  --manifest data/episodes/manifest.jsonl \
  --checkpoint checkpoints/best.pt \
  --out predictions.jsonl \
  --split val
```

## End-to-end на текущих данных FlowMetric

Полный демонстрационный прогон:

```bash
flowmetric-temporal current-data \
  --flowmetric-root /Users/alkselsv/Projects/FlowMetric \
  --out runs/current-data \
  --device auto \
  --epochs 30
```

Результаты:

```text
runs/current-data/
  annotations.jsonl       GT и weak-label эпизоды с аудитом
  episodes/manifest.jsonl извлечённый датасет
  episodes/*.npz          crops, trajectories и masks
  best.pt                 лучший checkpoint
  val_predictions.jsonl   предсказания validation split
  run.json                параметры и сводка запуска
```

Если сначала требуется просмотреть разметку без обучения:

```bash
flowmetric-temporal bootstrap-current \
  --flowmetric-root /Users/alkselsv/Projects/FlowMetric \
  --out runs/current-data/annotations.jsonl

# Либо сразу извлечь crops, но не обучать модель:
flowmetric-temporal current-data \
  --flowmetric-root /Users/alkselsv/Projects/FlowMetric \
  --out runs/current-data \
  --prepare-only
```

В текущем benchmark только две наблюдаемые положительные service-пары. Поэтому
`service` получает настоящие GT-метки, а остальные классы — геометрические
weak labels с `review_required: true`, `weak_label_scores` и `window_stats`.
Они предназначены для bootstrap и ручной проверки, а не считаются настоящим
ground truth. Поскольку все эпизоды происходят из одного 15-секундного ролика,
validation-метрики этого демонстрационного запуска не оценивают обобщение.

## Разметка hard negatives

Отрицательные случаи не следует объединять в один случайный фон. Их нужно
сохранять как ближайший содержательный класс и добавлять `service`-похожие
эпизоды в приоритетном порядке:

| Сцена | Метка |
|---|---|
| сотрудник проходит между столами | `passing` |
| сотрудник стоит рядом без взаимодействия | `waiting` |
| сотрудник остаётся после завершения обслуживания | `waiting` или отдельный следующий эпизод `cleanup` |
| уборка посуды/стола без обслуживания клиента | `cleanup` |
| обслуживание соседнего клиента | для рассматриваемой пары обычно `waiting`; также создать положительный эпизод с правильной парой |

Разговор с коллегой без клиента не соответствует контракту парного эпизода и
должен отбрасываться генератором кандидатов. Если такие сцены нужно явно
классифицировать, следующий осознанный шаг — добавить класс `other`, обновив
версию label schema.

## Ближайшие этапы

- инструмент просмотра и разметки кандидатов;
- pretraining визуального encoder и аугментации;
- hard-negative sampler по типу сцены и камере;
- интеграционный адаптер `ServiceVerifier` для основного FlowMetric;
- калибровка confidence и отдельный порог отказа от решения.
