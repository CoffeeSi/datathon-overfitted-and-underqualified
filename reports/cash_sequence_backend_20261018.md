# Подключение обученной cash-sequence модели к backend

Активный локальный сервис на 8001 переключён на `local-cash-sequence`, Docker frontend
на 3000 обращается к нему. Новый запуск по умолчанию:

```powershell
.\scripts\start-backend.ps1 -Port 8001
```

Python launcher и PowerShell wrapper выбирают `configs/action_service_cash_sequence.yaml`.
API key остаётся общим через `.env.compose`; mock не включается.

## Модели и семантика

- Двухракурсный R3D-18: `runs/action_improvement_20261018/primitives/last.pt`.
- Tiny U-Net: `runs/action_improvement_20261018/banknotes_stage2/last.pt`.
- Прежний бинарный checkpoint из `mild_augmented/last.pt` используется только
  для настоящего вспомогательного `binary_score`, совместимого с существующим API.
  Этот score не задаёт тип, время или допуск события в результат.
- Конфиг проверяет SHA-256 всех трёх checkpoint, loader — архитектуры и manifest;
  binary/primitive/detector должны иметь один split. Восстановление без скачивания pretrained.
- Клиент передал деньги → ушёл → купюры сохраняются на столе. Время берётся из
  причинной последовательности изображения, без поиска ID/меток в CSV.
- Выдаётся только `unsecured_accepted_cash`. Полной цепочки нет → `insufficient_data`,
  а не доказанное отсутствие нарушений. На каждый вызов создаётся новая temporal state machine.
- `type_confidence` — минимум оценок визуальных компонентов и свежей проверки изображения;
  это некалиброванный score. Ссылка из каталога условная; `legal_conclusion=null`.

## Проверка через настоящий frontend BFF

Числовые ответы: [cash_sequence_backend_20261018.json](cash_sequence_backend_20261018.json).
Действительные MOV загружены через `POST /api/review/analyze`, затем прочитаны
через `/api/review/jobs/{id}`. Это реальные модели, не синтетический pipeline.

- `/readyz` и frontend service: 200, provider `local-cash-sequence`, demoEnabled=false.
- X-2: accepted → done / insufficient_data, 0 событий, около 7.8 s.
- X-38: accepted → done / violation, `unsecured_accepted_cash` @29 s, 3 валидных
  JPEG кадра и ссылка на №231; около 19.6 s. Вспомогательный binary score 0.459
  не блокирует событие; sequence score 0.709.
- X-31: accepted → done / insufficient_data, 0 событий, около 7.8 s.
- Проверена неизменная jury-проекция `/api/v1/jobs/{id}`: только прежняя схема,
  без provider/recognition_scope и остальных дополнительных UI-полей.

Времена включают HTTP/очередь/инференс/кадры и не являются отдельным GPU benchmark.
JSON отчёта не содержит ключей или изображений; кадры сохраняются только в закрытом
локальном хранилище backend. Split, events и прежние веса сохранены.

## Интерфейс и ограничения

На странице нового результата показана область анализа одного сценария. Старые
сохранённые результаты `local-r3d18-actions` не переписываются: UI показывает
«Результат прежней модели» и ссылку для новой загрузки. История recent jobs продолжает
хранить только метаданные; новые job_id автоматически открываются с новым provider.

Оценка модели остаётся [сравнением на четырёх просмотренных development-видео](action_improvement_20261018.md)
с одним положительным: F1=1 относится только к визуальному сценарию. Это не
независимая оценка всех типов и не юридическое заключение. Ни личность/принадлежность
денег, ни завершение учёта/применимость нормы этим pipeline не установлены.

Проверки включают 156 Python-тестов, 84 frontend unit tests и production Docker build.
Тесты покрывают несколько событий с разными временами, отсутствие полной цепочки,
неподдерживаемые типы, сохранность файла, контроль hash, отмену и старые API-контракты.
Git commit и push в этой задаче не выполнялись.
