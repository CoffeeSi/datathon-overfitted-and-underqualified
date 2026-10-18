# Покрытие action-типов и готовность проверки качества

Источники: 42 ID, 40 канонических видео; train/validation 30/10. Split не изменён.

Новых реальных сеансов нет. Независимый test не создан и качество на нём не заявляется. Прежние метрики не пересчитаны и не заменены. Это аудит данных, а не обучение модели.

Перечисленные ниже положительные метки взяты из существующих action_review. Пакетные метки не объявляются независимым просмотром; отрицательное окно не доказывает отсутствие действия во всём видео. Каноническое видео считается один раз.

## unsecured_accepted_cash

- Train положительные: X-4; окон: 2; отрицательных окон: 22.
- Validation положительные: X-38; отрицательных окон: 3; подтверждённых отрицательных видео целиком: 0.
- Ограничения: single_canonical_train_positive, no_whole_video_negative_attestation, train_positives_include_batch_labels.
- План сбора: ещё 4 train-положительных и 1 validation-положительных исходных видео. Это ориентир для сбора, не обещание достаточной статистики.

## cash_counted_without_client_present

- Train положительные: нет; окон: 0; отрицательных окон: 24.
- Validation положительные: X-34; отрицательных окон: 6; подтверждённых отрицательных видео целиком: 0.
- Ограничения: no_train_positive_video, no_whole_video_negative_attestation, validation_positives_include_batch_labels.
- План сбора: ещё 5 train-положительных и 1 validation-положительных исходных видео. Это ориентир для сбора, не обещание достаточной статистики.

## cashier_leaves_during_client_service

- Train положительные: нет; окон: 0; отрицательных окон: 24.
- Validation положительные: нет; отрицательных окон: 7; подтверждённых отрицательных видео целиком: 0.
- Ограничения: no_train_positive_video, no_validation_positive_video, no_whole_video_negative_attestation.
- План сбора: ещё 5 train-положительных и 2 validation-положительных исходных видео. Это ориентир для сбора, не обещание достаточной статистики.

## client_identity_check_not_visible

- Train положительные: X-18; окон: 1; отрицательных окон: 23.
- Validation положительные: X-2; отрицательных окон: 6; подтверждённых отрицательных видео целиком: 0.
- Ограничения: single_canonical_train_positive, no_whole_video_negative_attestation, train_positives_include_batch_labels, validation_positives_include_batch_labels.
- План сбора: ещё 4 train-положительных и 1 validation-положительных исходных видео. Это ориентир для сбора, не обещание достаточной статистики.

## employee_personal_items_in_cash_unit

- Train положительные: X-40; окон: 1; отрицательных окон: 23.
- Validation положительные: нет; отрицательных окон: 7; подтверждённых отрицательных видео целиком: 0.
- Ограничения: single_canonical_train_positive, no_validation_positive_video, no_whole_video_negative_attestation, train_positives_include_batch_labels.
- План сбора: ещё 4 train-положительных и 2 validation-положительных исходных видео. Это ориентир для сбора, не обещание достаточной статистики.

## recount_without_visible_controller

- Train положительные: X-20; окон: 1; отрицательных окон: 23.
- Validation положительные: нет; отрицательных окон: 7; подтверждённых отрицательных видео целиком: 0.
- Ограничения: single_canonical_train_positive, no_validation_positive_video, no_whole_video_negative_attestation, train_positives_include_batch_labels.
- План сбора: ещё 4 train-положительных и 2 validation-положительных исходных видео. Это ориентир для сбора, не обещание достаточной статистики.

## collection_cash_amount_registration_not_observed

- Train положительные: X-13; окон: 1; отрицательных окон: 23.
- Validation положительные: нет; отрицательных окон: 7; подтверждённых отрицательных видео целиком: 0.
- Ограничения: single_canonical_train_positive, no_validation_positive_video, no_whole_video_negative_attestation, train_positives_include_batch_labels.
- План сбора: ещё 4 train-положительных и 2 validation-положительных исходных видео. Это ориентир для сбора, не обещание достаточной статистики.

## cash_bag_received_without_visible_identity_authority_check

- Train положительные: X-17; окон: 1; отрицательных окон: 23.
- Validation положительные: нет; отрицательных окон: 7; подтверждённых отрицательных видео целиком: 0.
- Ограничения: single_canonical_train_positive, no_validation_positive_video, no_whole_video_negative_attestation, train_positives_include_batch_labels.
- План сбора: ещё 4 train-положительных и 2 validation-положительных исходных видео. Это ориентир для сбора, не обещание достаточной статистики.

## safe_key_left_in_lock

- Train положительные: X-11, X-22; окон: 2; отрицательных окон: 22.
- Validation положительные: нет; отрицательных окон: 7; подтверждённых отрицательных видео целиком: 0.
- Ограничения: no_validation_positive_video, no_whole_video_negative_attestation, train_positives_include_batch_labels.
- План сбора: ещё 3 train-положительных и 2 validation-положительных исходных видео. Это ориентир для сбора, не обещание достаточной статистики.

## suspected_stamp_or_seal_left_unattended

- Train положительные: X-24; окон: 1; отрицательных окон: 23.
- Validation положительные: нет; отрицательных окон: 7; подтверждённых отрицательных видео целиком: 0.
- Ограничения: single_canonical_train_positive, no_validation_positive_video, no_whole_video_negative_attestation, train_positives_include_batch_labels.
- План сбора: ещё 4 train-положительных и 2 validation-положительных исходных видео. Это ориентир для сбора, не обещание достаточной статистики.

## Каталог и визуальное основание

Неоднозначные теги требуют контекста: X-15, X-26, X-38, X-44.
Ни один тег CSV сам по себе не подтверждает все визуальные условия. grounded_rule_match возвращает event_type и ссылки только при явных фактах и проверяемом происхождении доказательств. При неизвестном/ложном обязательном факте — event_type=null и source_links=[]. Даже разрешённая ссылка условна, legal_conclusion=null и requires_review=true.

Текущую редакцию официальных текстов повторно подтвердить не удалось: страница Әділет требует JavaScript. Сохранённый каталог и его хеш не менялись; №14/№53 не становятся новыми распознаваемыми типами.

## Дополнительные данные

Аугментаций: 168; синтетических файлов: 9. Независимых test-кандидатов: 0. Синтетические сценарии остаются отдельным потенциальным вспомогательным источником после проверки, без автоматического импорта их заявлений как фактов.

## Что остаётся открытым

Реальное добавление независимых исходных примеров и итоговый test требуют новых записей. Старый split не переразмечается ради получения метрики; модель с accuracy 60% не заменяется. Backend/frontend и конфигурация активной модели не изменялись.

Хеши входов: `{"catalog_sha256": "8ceabe739482859817436f53b203290e90092e43153d33a2e7f6a079377eb2fe", "events_sha256": "4332f2bbfdff79534ad98d881287137baba83cc4c47756d3607af409742ee42b", "split_sha256": "19c33f7e336b51fe5705f62d1804fb05e9c557076217aa5c0d6f986eb22f363d"}`.

## Выполненные проверки

139 Python-тестов прошли. Новая секция EDA исполнена в двух чистых kernels из корня и notebooks/. Outputs/counts/attachments очищены; git diff --check пройден. Хеши исходной разметки, split, каталога и сохранённой модели совпадают. Подробности — verification.json.
