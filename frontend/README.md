# Overfitted and Underqualified — просмотр кассового видео

Рабочий интерфейс ForteBank Datathon: загрузка `.mov`, ожидание задания,
просмотр результата, временная шкала, описание событий и подтверждающие кадры.
Next.js App Router, React, TypeScript и Tailwind; зависимости фиксирует `package-lock.json`.

## Docker Compose

### Brev: оба сервиса в Docker с NVIDIA GPU

Используйте оба файла: `compose.yaml` и `compose.real.yaml`. Базовый файл
рассчитан на ML на локальном хосте и сам не запускает backend по умолчанию.
Override включает backend, ML-зависимости, одну NVIDIA GPU, read-only `runs/`
и текущий `configs/action_service_cash_sequence.yaml`. Frontend обращается к
`http://backend:8000`; он стартует после успешного `/readyz` модели.

Создайте `.env.compose` с непустым `FORTEBANK_API_KEY`. Перенесите все три веса
по путям из конфига: они не входят в Git. Если в старом `.env.compose` задан
`ACTION_SERVICE_CONFIG`, замените его на `./configs/action_service_cash_sequence.yaml`.

```bash
docker compose --env-file .env.compose -f compose.yaml -f compose.real.yaml up --build -d
docker compose --env-file .env.compose -f compose.yaml -f compose.real.yaml ps
curl -i http://127.0.0.1:8000/readyz
```

Ожидается `local-cash-sequence`. Диагностика:
`docker compose --env-file .env.compose -f compose.yaml -f compose.real.yaml logs --tail=100 backend`.
Нужны NVIDIA GPU, совместимый драйвер и GPU runtime Docker; одним наличием GPU
в `nvidia-smi` на хосте её доступ внутри контейнера не подтверждается.

### Текущий запуск: настоящая ML-модель на Windows, frontend в Docker

Backend запускается из корня через `scripts/start-backend.ps1 -Port 8000`.
Общий `FORTEBANK_API_KEY` задаётся в `.env.compose`: Compose читает его через
`env_file`, а `scripts/start-backend.ps1` — через Python launcher. Значение из
файла имеет приоритет над устаревшим ключом в терминале. После изменения ключа
нужно перезапустить backend и пересоздать контейнер frontend (не только restart).
Для локального `npm run dev` тот же ключ указывается в `frontend/.env.local`,
с `BACKEND_URL=http://127.0.0.1:8000` и `DEMO_ENABLED=false`.
Файлы с ключами не коммитятся.

```powershell
.\scripts\start-backend.ps1 -Port 8000
# В другом терминале, из корня:
docker compose --env-file .env.compose -f compose.yaml -f compose.host-ml.yaml up -d --build --no-deps frontend
```

Docker frontend обращается к `http://host.docker.internal:8000`.
Это также адрес по умолчанию в базовом `compose.yaml`; обычный повторный запуск
не переключает frontend обратно на не настроенный контейнер backend.
Overrides `compose.real.yaml` и `compose.mock.yaml` явно выбирают контейнерный API.
Откройте `http://127.0.0.1:3000`; `/api/review/service` должен вернуть
`provider=local-cash-sequence`, `ready=true`, `demoEnabled=false`.
В этом режиме mock-провайдер блокируется перед загрузкой и при чтении старых
результатов; ошибки backend не заменяются синтетическими ответами.
Старые задания от mock не превращаются в ML-результаты после смены провайдера:
нужно заново загрузить исходное видео. Если доступен старый mock-результат,
API возвращает 410 с пояснением; автоматический опрос такого задания прекращается.
Активен `configs/action_service_cash_sequence.yaml`: распознавание клиента и передачи
денег, детектор купюр и временная цепочка. Только сценарий денег, оставшихся после
ухода клиента; остальные типы не поддержаны. Отсутствие полной цепочки возвращает
`insufficient_data`. Область анализа показана в интерфейсе. Старые задания
`local-r3d18-actions` сохраняют прежние результаты; для нового анализа нужна новая загрузка.
Измеренная приёмка охватывает лишь 4 просмотренных development-видео, не независимый test.
См. [проверку реального backend](../reports/cash_sequence_backend_20261018.md).

### Явный mock для разработки интерфейса

Из корня репозитория скопируйте `.env.compose.example` в `.env.compose`, задайте
непустой `FORTEBANK_API_KEY` и запустите:

```powershell
Copy-Item .env.compose.example .env.compose
# Задайте FORTEBANK_API_KEY в .env.compose
docker compose --env-file .env.compose -f compose.yaml -f compose.mock.yaml up --build -d
```

Откройте [http://127.0.0.1:3000](http://127.0.0.1:3000). Проверка соединения:

```powershell
Invoke-RestMethod http://127.0.0.1:3000/api/review/service
```

Ожидаются `health=online`, `ready=true`, `provider=mock`. Контейнеры используют
Next.js BFF и FastAPI в одной Docker-сети; API также доступен локально на порту
8000. Данные заданий сохраняются в Docker volume `backend-data`. Этот запуск
явно использует mock analyzer: он проверяет загрузку и получение результата,
но не распознаёт содержимое видео. Для остановки используйте
`docker compose --env-file .env.compose down`; volume при этом сохраняется.

Для текущего R3D-18 action analyzer предусмотрен отдельный Compose override.
Нужны **три локальных checkpoint** (primitives, banknotes и auxiliary binary), совместимый конфиг с
путями к ним и хотя бы один оценённый тип в `enabled_types`. Скопируйте
`configs/action_service_cash_sequence.yaml` в `configs/action_service.local.yaml`, укажите
пути относительно корня проекта внутри контейнера (`runs/...`), проверенные
типы и `device: cpu`. Этот локальный конфиг игнорируется Git. Укажите в
`.env.compose` `MODEL_RUNS_PATH=./runs` (или полный путь к вашему каталогу runs)
и `ACTION_SERVICE_CONFIG=./configs/action_service.local.yaml`, затем запустите:

```powershell
docker compose --env-file .env.compose -f compose.yaml -f compose.real.yaml up --build -d
```

Override устанавливает ML-зависимости, монтирует веса только для чтения и
выключает mock без автоматической подмены. Проверьте `/readyz`: ожидается
`analysis_provider=local-cash-sequence`. При отсутствии файлов, несовместимых
весах или пустом `enabled_types` readiness останется 503; причина будет в
`docker compose logs backend`. Веса не входят в Git; конфиг по умолчанию
содержит `enabled_types: []`. Для текущего cash-sequence варианта используйте
пути, SHA-256 и типы из `configs/action_service_cash_sequence.yaml`; на CPU явно замените
`device: cuda` на `device: cpu` в своём локальном конфиге. Качество action-типов
не подтверждено. См. [план интеграции](../docs/frontend-backend-wiring-plan.md)
и [оценку модели](../reports/action_recognition_status.md).

## Локальный запуск

Используйте Node.js 24 (проверено на 24.14.0) или 22.13+, npm и отдельное
Python-окружение backend. Команды ниже выполняются в PowerShell.

Из корня репозитория запустите API:

```powershell
.\scripts\start-backend.ps1 -Port 8000
```

В другом терминале:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env.local
```

В `frontend/.env.local` укажите **тот же** `FORTEBANK_API_KEY`, что задан backend.
Frontend не читает автоматически `.env` из корня Python-проекта.

```powershell
npm run dev -- --hostname 127.0.0.1
```

Откройте [http://127.0.0.1:3000](http://127.0.0.1:3000). На Linux команды npm те же;
для копирования используйте `cp .env.example .env.local`, а переменные Python-процесса
задавайте через `export`.

Без настроенного Analyzer API запускается, но `/readyz` отвечает `503`.
Интерфейс показывает, что анализатор пока не подключён; доступен выбор и локальный
предпросмотр файла. Это ожидаемое состояние текущего проекта.

## Конфигурация

Все переменные находятся только на сервере Next.js:

- `BACKEND_URL`: origin API, по умолчанию `http://127.0.0.1:8000`; без пути,
  query, логина и пароля. Допускаются HTTP и HTTPS.
- `FORTEBANK_API_KEY`: общий секрет для запросов к защищённым endpoint API.
  Если пуст, реальная отправка и чтение задания недоступны. Неверный ключ приводит к `401`.
- `MAX_UPLOAD_MB`: целое число от 1 до 10240, по умолчанию 100; согласуйте с
  `FORTEBANK_MAX_UPLOAD_MB` backend и лимитом reverse proxy/хостинга.
- `DEMO_ENABLED`: только `true` включает синтетические примеры; по умолчанию `false`.

Не помещайте ключ в `NEXT_PUBLIC_*`, исходники, URL или браузерное хранилище.
После изменения серверной конфигурации перезапустите Next.js.
`.env.local` игнорируется Git; `.env.example` содержит только безопасный шаблон.

## Явный демонстрационный режим

Установите `DEMO_ENABLED=true` в `.env.local` и перезапустите frontend.
Примеры доступны без работающего Python API и явно помечены как демонстрационные:

- `/review/demo-violation`: три синтетических события, в том числе близкие отметки,
  длинное описание и новый неизвестный тип.
- `/review/demo-processing`: ожидание, затем демонстрационный результат
  после двух секунд наблюдения страницы.
- `/review/demo-no-violation`: результат без нарушений.
- `/review/demo-insufficient`: недостаточно данных.
- `/review/demo-failed`: завершение с ошибкой.

Статические JPEG в `public/demo/` — нарисованные схемы, а не банковские кадры.
Provider `synthetic-demo` не обращается к модели и никогда не подменяет ошибку
реального запроса успешным результатом. При выключенном режиме эти ID возвращают `404`.

Отдельный режим backend `mock` тоже явно выделяется интерфейсом. Он включается
на стороне Python через `FORTEBANK_ENABLE_MOCK_RESULTS` и, при необходимости,
`FORTEBANK_MOCK_FIXTURE_PATH`. Загрузка в mock API действительно передаёт файл
backend; синтетические ссылки выше ничего не загружают.

## API и поведение

Браузер обращается только к same-origin BFF:

- `GET /api/review/service` запрашивает `/healthz` и `/readyz`, возвращает состояние
  соединения, готовность, provider, лимит загрузки и доступность демо.
- `POST /api/review/analyze` передаёт multipart с единственным полем `file`
  в `POST /api/v1/analyze`. Ответ `202` содержит `job_id`.
- `GET /api/review/jobs/{jobId}` получает `GET /api/v1/ui/jobs/{jobId}`.
  UI-версия сохраняет `analysis_provider` и безопасную ошибку failed-задания.

Ключ добавляется BFF на сервере. У upstream фиксированный набор путей;
редиректы не выполняются. Ответы приватные, `Cache-Control: private, no-store`.
Транспорт проверяет статусы, verdict, конечные неотрицательные времена,
форму findings, 1–3 JPEG evidence frames и соответствие `violation` непустому списку.
Неизвестный тип события остаётся доступен как исходный тип и описание.
Повреждённый кадр имеет отдельное UI-состояние.

Загрузка проходит потоком `Request.body → TransformStream → fetch`, без
`formData()` и полного копирования видео в памяти Next.js. Поток ограничен лимитом
файла плюс 1 MiB на multipart. Backend дополнительно проверяет точный размер
файла, единственное поле `file`, расширение и MOV-контейнер. POST требует тот же
Origin, что origin страницы приложения; при ручных HTTP-проверках его нужно указать явно.

Процент относится к передаче файла из браузера в BFF. 100% не означает завершение
анализа или подтверждённое принятие backend. Пока нет ответа `202`, интерфейс
ожидает подтверждения. POST автоматически не повторяется: после сетевого обрыва
или timeout сервис мог принять файл, а новая отправка создаст другое задание.

Polling работает последовательно, останавливается на `done`/`failed`, отменяется
при смене задания и скрытой вкладке, возобновляется при возвращении. Временные
ошибки чтения получают backoff; `401`/`403`/`404` требуют явного действия.
После пяти минут наблюдения автоматическая проверка приостанавливается; задание
может продолжить работу на сервере. Счётчик отражает наблюдение в этой вкладке,
а не неизвестное время работы модели.

Таймауты BFF: health/readiness — 4 секунды, чтение задания — 12 секунд,
приём загрузки — 180 секунд. Браузер ждёт ответ загрузки до 185 секунд.
Хостинг должен поддерживать Node.js route handlers, потоковую передачу и выбранные
размеры/таймауты; для небольших serverless upload-лимитов потребуется другая
согласованная схема загрузки, а не только повышение `MAX_UPLOAD_MB`.

## Просмотр, локальные данные и ограничения

`/` содержит загрузку и недавние задания. `/review/[jobId]` можно открыть напрямую
или восстановить по ID. Выбор события связан со списком, временной шкалой,
описанием и кадрами. При доступном локальном видео выбор переводит плеер к отметке.
JSON скачивается только по действию пользователя.

История «На этом устройстве» сохраняет в localStorage не больше 20 записей:
`job_id`, имя файла, время создания и последнее известное состояние. Исходные видео,
полные результаты, base64 кадров и ключи там не хранятся. Историю можно очистить;
это не удаляет задания backend. В среде без localStorage список работает в памяти.

Выбранный видеофайл остаётся в памяти текущей вкладки. После перезагрузки результат
и кадры запрашиваются из API, а для воспроизведения нужно снова выбрать исходный файл.
Backend не возвращает его URL или хеш, поэтому соответствие повторно выбранного
файла заданию нельзя подтвердить автоматически. Некоторые MOV-кодеки не поддерживаются
браузером: ошибка preview не запрещает отправку корректного файла на анализ.

Срок хранения задаётся backend (по умолчанию 24 часа). Неизвестные и истёкшие задания
возвращают `404`; локальная история не гарантирует наличие результата на сервере.

Текущий R3D-18 scanner возвращает **бинарные кандидаты без типа действия**. Он ещё
не является типизированным `Analyzer`, подходящим к публичному `Finding`.
Frontend не назначает статью закона по бинарному verdict, не выдумывает confidence
и не превращает отсутствие кадров в доказательство отсутствия действия.
В текущем API Finding нет нормативных ссылок; они не синтезируются интерфейсом.
Граница интеграции описана в [full-video-inference.md](../docs/full-video-inference.md).

## Доступ и приватность

Проект предназначен для локального или закрытого datathon-демо. BFF скрывает общий
backend key и проверяет origin POST, но **не авторизует пользователя**. У endpoints
нет пользовательских аккаунтов, проверки владельца задания или разграничения доступа.
Не публикуйте BFF с реальными видео в открытый интернет без отдельного слоя доступа,
TLS и согласованных ограничений сети/прокси. Для локального запуска используйте
`--hostname 127.0.0.1`, как в инструкции выше.

Видео и evidence не отправляются в сторонние AI/CDN/image-optimization сервисы.
Банковские кадры не входят в `public/`, fixtures или Git. Скриншоты реальной сессии
и скачанные JSON могут содержать чувствительные данные; храните их локально.
Публичные demo assets содержат только синтетические схемы.

## Структура и проверки

- `app/`: маршруты, layout, страницы и server-only BFF.
- `features/upload/`: выбор, предпросмотр, проверка и отправка файла.
- `features/analysis-job/`: жизненный цикл задания и polling.
- `features/review/`: presentation model, timeline, просмотр событий и кадров.
- `features/recent-jobs/`: ограниченная локальная история метаданных.
- `components/ui/`: общие доступные элементы интерфейса.
- `lib/contracts.ts`: wire types и runtime validation.
- `lib/client.ts`: браузерный транспорт; `lib/server/`: конфигурация, upstream и демо.
- `tests/`: контракты, сетевые границы и поведение сценариев.

```powershell
npm run lint
npm run typecheck
npm test
npm run build
npm run start -- --hostname 127.0.0.1
```

### Браузерные проверки Playwright

Установите браузер для тестов:

```powershell
npx playwright install chromium
npm run test:e2e
```

Вместо загрузки Chromium можно использовать уже установленный Google Chrome:

```powershell
$env:PW_BROWSER_CHANNEL = "chrome"
npm run test:e2e
```

Если frontend ещё не запущен, Playwright поднимает dev server на `127.0.0.1:3000`
с явно установленным `DEMO_ENABLED=true`. Если локальный сервер уже работает,
тесты используют его: включите demo самостоятельно и перезапустите сервер.
Синтетические сценарии не требуют backend. Проверка неизвестного задания пропускается,
если FastAPI недоступен.

Опциональная проверка настоящей HTTP-загрузки требует явно включённого **mock** backend,
совпадающих ключей и безопасного MOV длительностью не меньше 5 секунд, который браузер
может воспроизвести. В `FORTEBANK_MOCK_FIXTURE_PATH` задайте валидный JSON AnalysisResult:
`verdict=violation`, два Finding на 2-й и 4-й секунде, каждый с 1–3 синтетическими JPEG
в `evidence_frames`. Тест ожидает второй таймкод 4 секунды.

```powershell
$env:E2E_MOV_PATH = "D:\absolute\path\synthetic-review.mov"
npm run test:e2e
```

Без `E2E_MOV_PATH` этот интеграционный тест пропускается. Не используйте банковские
записи для тестовых screenshots/traces. `test-results/` и `playwright-report/`
содержат локальные артефакты тестов и исключены из Git.
Измеренные HTTP-проверки текущей реализации описаны в [VERIFICATION.md](VERIFICATION.md).

Для ручной проверки пройдите upload, processing, три вида результата, failed,
backend offline и provider unavailable. Проверьте прямую ссылку на задание,
поиск и выбор событий, близкие маркеры, увеличение кадра, клавиатурную навигацию,
узкий экран и уменьшение анимаций через `prefers-reduced-motion`.

Автоматические проверки используют синтетические данные. Успешная проверка демо
подтверждает работу интерфейса, а не качество реальной модели.
