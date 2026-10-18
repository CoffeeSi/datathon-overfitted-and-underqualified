# Action-модель: обучение и полный инференс, 18 октября 2026

Два реальных запуска по пять эпох на CUDA. 31 подтверждённое окно: 24 train и 7 validation. Обучались восемь типов; только client_identity_check_not_visible имеет положительный и отрицательные validation-примеры. Подтверждение владельца записано в action_review того же events.csv. Неоднозначные строки masked, все прежние поля, времена и split сохранены. Никакое отсутствие не распространено на полное видео.

Первый запуск использует обычную masked BCE. Второй — балансировку positive loss отношением известных отрицательных к положительным train-примерам; validation для расчёта весов не используется. Порог 0.7, фон 0.2, пять эпох и временной допуск 5 секунд зафиксированы; порог не подбирался. Второй запуск — исследовательская итерация после первого; независимого теста нет.

## none

Checkpoint: `runs/r3d18_actions_owner_20261018T004011_430311Z/last.pt`; SHA-256 `1232cbcf29b8db65f4bf89f9131491f7927e34a106847d3eefd4dec6be6954cb`.

Отчёт и ZIP: `reports/evaluation_actions_20261018T004341Z_efe9e4b9` / `reports/evaluation_actions_20261018T004341Z_efe9e4b9.zip`.

client_identity_check_not_visible: precision=0.000, recall=0.000, F1=0.000; confusion [[TN,FP],[FN,TP]] = [[6, 0], [1, 0]]; support=1.

Полные 10 validation-видео: 102.91 с обработки на CUDA; бинарный фильтр пропустил 115 окон, тип выдан в 0. Это wall time данного запуска, а не нормированный benchmark.

Ошибка FN: X-2 @ 10.00 с, client_identity_check_not_visible, score=0.0555.

Полный инференс cash_counted_without_client_present: matched=0, missed=1, binary gate coverage=1.0, MAE=None.

Полный инференс client_identity_check_not_visible: matched=0, missed=1, binary gate coverage=1.0, MAE=None.

## train_ratio

Checkpoint: `runs/r3d18_actions_balanced_20261018T004011Z/last.pt`; SHA-256 `4c307821390f6608d71a0f658b1875db72bdb2244330a510d22a843f035363ba`.

Отчёт и ZIP: `reports/evaluation_actions_20261018T004537Z_34962505` / `reports/evaluation_actions_20261018T004537Z_34962505.zip`.

client_identity_check_not_visible: precision=0.000, recall=0.000, F1=0.000; confusion [[TN,FP],[FN,TP]] = [[6, 0], [1, 0]]; support=1.

Полные 10 validation-видео: 94.32 с обработки на CUDA; бинарный фильтр пропустил 115 окон, тип выдан в 0. Это wall time данного запуска, а не нормированный benchmark.

Ошибка FN: X-2 @ 10.00 с, client_identity_check_not_visible, score=0.2471.

Полный инференс cash_counted_without_client_present: matched=0, missed=1, binary gate coverage=1.0, MAE=None.

Полный инференс client_identity_check_not_visible: matched=0, missed=1, binary gate coverage=1.0, MAE=None.

## Вывод и ограничения

Обучение и восстановление реальных весов выполнены. 120 тензоров backbone, включая BatchNorm, сверены с исходными pretrained-весами и не изменились. Тесты проверяют отсутствие обновления на validation, несколько событий, пустой результат, неизвестные типы и привязку правил. Синтетические проверки не доказывают качество распознавания банковских событий.

Измеренное качество недостаточно: единственный оцениваемый положительный пример пропущен. Другие семь train-типов не имеют положительных validation-примеров. Ещё один тип есть только в validation и считается пропуском, а не используется для обучения. Пропуски и binary gate coverage разделены в JSON. При отсутствии совпадений MAE=null. Лишние тревоги на неполной временной разметке не объявляются подтверждёнными FP.

Общий кабинет, люди и неизвестные сеансы могут пересекать выборки. Разметка подтверждена владельцем пакетно, не прошла независимую проверку. В отчёты и ZIP не включены кадры или веса.

В основном сервисе enabled_types остаётся пустым. Для воспроизведения используется configs/action_service_research.yaml; это экспериментальный режим с известными пропусками. Для закрытия P0 по распознаванию нужны дополнительные положительные видео в обеих выборках и разбор X-2/контекста идентификации: одна обучающая точка не подтверждает обобщение.
