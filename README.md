# Video Analytics Pipeline

[![CI](https://github.com/NTsundere/video-analytics-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/NTsundere/video-analytics-pipeline/actions/workflows/ci.yml)

Учебный проект: обработка видеопотоков в реальном времени. Камера (файл или RTSP) читается в отдельном потоке, кадры уходят в сервис инференса (SSDLite, PyTorch), результат с рамками и идентификаторами объектов раздаётся клиентам по WebSocket, новые объекты пишутся в PostgreSQL. Стек: FastAPI, asyncio, SQLAlchemy 2.x, OpenCV, Docker Compose.

![demo](docs/demo.gif)

## Архитектура

```
video file / RTSP -> FrameGrabber (thread) -> CameraWorker (asyncio) -> Inference Service (HTTP, JPEG)
                                                    |                          |
                                                    |<------ detections -------+
                                                    +--> IoU tracker --> new objects --> EventWriter (queue, batches) --> PostgreSQL
                                                    +--> annotated JPEG --> N WebSocket clients
```

- **api** — FastAPI. REST для камер, событий и метрик, WebSocket `/ws/stream/{camera_id}`, раздача `frontend/`.
- **inference** — отдельный сервис с моделью. `/detect/{binary,base64,raw}` и `/echo/{binary,base64,raw}` (последние только декодируют кадр, нужны для бенчмарка транспорта).
- **db** — PostgreSQL 16.

## Запуск

```bash
cp .env.example .env
cp /path/to/videos/*.mp4 videos/
docker compose up --build
```

Откройте http://localhost:8000. Для каждого файла из `videos/` при старте создаётся камера. Swagger: http://localhost:8000/docs

### RTSP

```bash
cp /path/to/video.mp4 videos/demo.mp4
docker compose -f docker-compose.yml -f docker-compose.rtsp.yml up --build

curl -X POST http://localhost:8000/api/cameras \
  -H "Content-Type: application/json" \
  -d '{"name": "rtsp-demo", "source": "rtsp://mediamtx:8554/demo"}'
```

Если остановить контейнер `rtsp-feed`, воркер переходит в режим переподключения и возобновляет поток, когда источник вернётся.

### Тесты и линтер

```bash
cd api
pip install -r requirements-dev.txt
pytest
cd ..
ruff check .
```

### Бенчмарк

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d
pip install -r benchmark/requirements.txt
python benchmark/benchmark.py --video videos/<file>.mp4
```

Результаты запуска:

```
frame 640x360, runs=50, warmup=5

mode            payload KB   echo p50   echo p95  detect p50  detect p95  detect mean
-------------------------------------------------------------------------------------
raw ndarray          675.0        3.1       49.4        58.4       107.4         64.4
jpeg binary           13.0       50.0       59.4        60.2        74.0         61.6
jpeg base64           17.4       50.1       59.7        63.7        69.7         64.0

transport+decode p50, base64 vs binary: +0.1 ms
transport+decode p50, raw vs binary:    -46.8 ms
payload, base64 vs binary:              +33.4%
payload, raw vs binary:                 51.8x
```

## Метрики

`GET /api/metrics` отдаёт по каждому воркеру скользящее среднее времени стадий (подготовка кадра, инференс с учётом HTTP, отрисовка, итого), FPS обработки, число обработанных и пропущенных кадров, ошибки инференса и число зрителей. Те же данные показаны на странице справа.

## Дизайн-решения

1. **Чтение кадров не блокирует event loop.** `cv2.VideoCapture.read` блокирующий, поэтому он работает в отдельном потоке `FrameGrabber`. Тяжёлые CPU-операции кадра (resize, JPEG-кодирование, отрисовка) выполняются через `asyncio.to_thread`.
2. **Только последний кадр.** Grabber хранит самый свежий кадр, воркер берёт его, когда освободился. Если инференс медленнее камеры, кадры пропускаются, а задержка не накапливается. Количество пропущенных кадров видно в метриках. Видеофайл воспроизводится в реальном времени по FPS файла и зацикливается.
3. **Инференс в отдельном сервисе.** Основная причина — независимое масштабирование и изоляция зависимостей и памяти. GIL здесь не главный фактор: PyTorch и OpenCV отпускают его в нативном коде. Число параллельных инференсов ограничено семафором, число потоков torch задаётся переменной окружения.
4. **Передача кадров.** Кадр уменьшается до ширины 640 с сохранением пропорций и кодируется в JPEG (quality 80), координаты рамок масштабируются обратно. Выбор между raw / binary / base64 подкреплён бенчмарком.
5. **Fan-out.** На одну камеру создаётся один воркер, кадры раздаются всем подписчикам. Воркер запускается при первом подписчике и останавливается, когда ушёл последний.
6. **Запись событий вне критического пути.** Воркер кладёт события в ограниченную очередь, `EventWriter` пишет их батчами. Если БД недоступна, стрим продолжает работать. При переполнении очереди события отбрасываются и считаются.
7. **Трекинг.** Простой IoU-трекер: детекции сопоставляются с существующими треками того же класса, событие записывается один раз, когда объект подтверждён несколькими кадрами подряд. Это не замена полноценным трекерам (SORT, ByteTrack): нет предсказания движения, нет повторной идентификации после долгой потери.
8. **Безопасность источников.** Через API можно добавить только `rtsp://` и `rtsps://`. Файловые камеры создаются только из каталога `videos/`, путь проверяется на выход за его пределы. API рассчитан на доверенного пользователя: проверки RTSP-адресов на внутренние сети нет.

## Ограничения и следующие шаги

- Таблицы создаются через `create_all`, миграций (Alembic) нет.
- Кадр обрабатывается последовательно (подготовка, инференс, отрисовка). Возможное улучшение — разделить стадии очередью размера 1.
- Модель SSDLite выбрана ради скорости на CPU, качество ниже, чем у YOLO-семейства. Следующий шаг — ONNX Runtime.
- Аутентификации нет.
- Число одновременных воркеров ограничено `MAX_WORKERS` (по умолчанию 4).

## Структура

```
api/            FastAPI-приложение и тесты
inference/      сервис инференса
frontend/       страница просмотра
benchmark/      сравнение способов передачи кадров
docs/           скриншоты для README
videos/         видеофайлы (не в git)
.github/        CI
```
