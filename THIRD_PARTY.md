# Используемые open-source компоненты

Собственные исходники подготовлены командой в этом проекте; чужие публичные
решения кандидатов не копировались.

| Компонент | Назначение | Лицензия |
|---|---|---|
| intfloat/multilingual-e5-base | Семантические embeddings, без дообучения | MIT |
| NumPy | Численные массивы, fp16/fp32 и cosine | BSD-3-Clause |
| Polars | Parquet/CSV, join, группировки и сортировки | MIT |
| LightGBM | Три binary-модели ранжирования | MIT |
| Narwhals | Внутренняя зависимость LightGBM для таблиц | MIT |
| snowballstemmer | Русский stemming | BSD-3-Clause |
| SciPy | Sparse-операции исходного lexical retrieval | BSD-3-Clause |
| scikit-learn | CountVectorizer исходного BM25 | BSD-3-Clause |
| PyTorch | Исходное GPU-кодирование и dense retrieval | BSD-3-Clause |
| Transformers | Исходная загрузка e5 | Apache-2.0 |
| Sentence Transformers | Исходное кодирование e5 | Apache-2.0 |

Точная ревизия e5: `d128750597153bb5987e10b1c3493a34e5a4502a`.
Модель: https://huggingface.co/intfloat/multilingual-e5-base
Версии исходной среды: `provenance/environment.json`.
Зависимости финального CPU-инференса закреплены
в `requirements.txt`; neural framework в обычном запуске не импортируется.

Готовые модельные веса/embeddings применяются локально. Runtime не выполняет
внешние API-вызовы. Загрузка библиотек при подготовке окружения не является
частью вычисления ответа. Если заново строить embeddings, нужен локальный
checkpoint закреплённой модели и исходная neural-среда; для поставленного
файла повторное кодирование не нужно.

Тексты лицензий установленных библиотек сохранены в `third_party_licenses/`.
