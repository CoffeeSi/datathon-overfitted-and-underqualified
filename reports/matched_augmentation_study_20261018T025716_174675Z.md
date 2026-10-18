# Сравнение оригиналов и мягкой смеси аугментаций

Обе модели обучены с одинаковой начальной головой и ровно 120 обновлениями. Validation не выбирала эпоху, checkpoint или порог; использованы фиксированные последние веса.

Это бинарная классификация окон вокруг известных отметок. 40 преобразований относятся к10 источникам и не являются независимым test.

## control

Веса: `D:\CodeSpace\datathon\datathon-overfitted-and-underqualified\runs\r3d18_matched_20261018T025716_174675Z\control\last.pt`.
SHA-256: `dab3376c6940c55f34ffd667a4aa538efb6631231ada69c958aadf05b4a7df3e`.

- original_video_metrics: accuracy=0.600, precision=0.667, recall=0.400, F1=0.500; confusion=[[4, 1], [3, 2]].
- transformed_video_metrics: accuracy=0.650, precision=0.800, recall=0.400, F1=0.533; confusion=[[18, 2], [12, 8]].

Результаты по преобразованиям:

- cctv_low_quality: accuracy=0.700, F1=0.571.
- cool_camera: accuracy=0.700, F1=0.571.
- mirror: accuracy=0.600, F1=0.500.
- warm_sepia: accuracy=0.600, F1=0.500.

## mild_augmented

Веса: `D:\CodeSpace\datathon\datathon-overfitted-and-underqualified\runs\r3d18_matched_20261018T025716_174675Z\mild_augmented\last.pt`.
SHA-256: `00b07713d0fd1c219fcefa00416f6e68668201ee8abec0e8b6bd43682dab00d9`.

- original_video_metrics: accuracy=0.600, precision=0.667, recall=0.400, F1=0.500; confusion=[[4, 1], [3, 2]].
- transformed_video_metrics: accuracy=0.650, precision=0.800, recall=0.400, F1=0.533; confusion=[[18, 2], [12, 8]].

Результаты по преобразованиям:

- cctv_low_quality: accuracy=0.700, F1=0.571.
- cool_camera: accuracy=0.700, F1=0.571.
- mirror: accuracy=0.600, F1=0.500.
- warm_sepia: accuracy=0.600, F1=0.500.

## Интерпретация

У мягкой смеси accuracy на 40 преобразованиях: 65.0%; контроль: 65.0%.
На 10 оригиналах обе модели имеют accuracy=60% и F1=0.50. На 40 вариантах
F1=0.533, Precision=0.80, Recall=0.40: TN=18, FP=2, FN=12, TP=8.
100% не достигнуты. В этом контролируемом сравнении преимущество аугментаций
не подтверждено: обе модели дали одинаковые классы при фиксированном пороге.
Это выше предыдущего неудачного запуска на всех аугментациях (accuracy62.5%,
F1=0.40), но ниже исторического ROI checkpoint (accuracy70%, F1=0.667).
Последнее сравнение историческое: прежние веса сейчас отсутствуют, а протокол
обучения отличался. Новые данные не объявляются решением проблемы обобщения.
Результат 100% не гарантируется. Неизменность backbone/BatchNorm, одинаковая инициализация и количество шагов сохранены в checks.json каждого запуска. Исходные метки и split сохранены. Порог 0.5 не подгонялся к validation.
Обученные прежние checkpoints перед этим запуском отсутствовали; выполнено свежее обучение от локального pretrained backbone. Исторические метрики не переписывались. Основной сервис автоматически не переключён.

См. [метод и повторение](../docs/matched-augmentation.md).

130 Python-тестов прошли; новые секции 02/03 выполнены в четырёх чистых kernels
из корня и notebooks/. Source outputs, execution counts и изображения очищены.
Хеши events.csv и split совпадают с началом эксперимента; git diff --check пройден.
Коммит и push не выполнялись.
