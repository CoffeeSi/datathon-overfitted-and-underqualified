# Catalog coverage for action-recognition research

This audit maps existing descriptive tags via event_type_for_observed(). It creates no training targets.
Counts use the agreed canonical split; duplicate sources are not independent examples.

- events_sha256: `2ec181a7af08257f882aeb006b6f98b5ccb0f204470adb92d252616a67b1e062`
- split_sha256: `19c33f7e336b51fe5705f62d1804fb05e9c557076217aa5c0d6f986eb22f363d`
- catalog_sha256: `431864b992042e9470b34eaf6690e066b4218dbe65693992daaa91318074f810`

| Supported type | Source rows | Canonical videos | Train | Validation | Distinct canonical markers |
| --- | ---: | ---: | ---: | ---: | ---: |
| unsecured_accepted_cash | 5 | 4 | 2 | 2 | 5 |
| cash_counted_without_client_present | 1 | 1 | 0 | 1 | 1 |
| cashier_leaves_during_client_service | 1 | 1 | 1 | 0 | 1 |
| client_identity_check_not_visible | 2 | 2 | 1 | 1 | 2 |
| employee_personal_items_in_cash_unit | 1 | 1 | 1 | 0 | 1 |
| recount_without_visible_controller | 1 | 1 | 1 | 0 | 1 |
| collection_cash_amount_registration_not_observed | 1 | 1 | 1 | 0 | 1 |
| cash_bag_received_without_visible_identity_authority_check | 1 | 1 | 1 | 0 | 1 |
| safe_key_left_in_lock | 2 | 2 | 2 | 0 | 2 |
| suspected_stamp_or_seal_left_unattended | 1 | 1 | 1 | 0 | 1 |

## Candidate for a separate experiment

Select **unsecured_accepted_cash** as a hypothesis for reviewing visible cash left at the workstation. It has the broadest coverage: four unique videos, two train and two validation, five point markers. X-4 and X-47 share one training video but retain two distinct markers. The other training example X-15 assumes cash inside a closed bag. Thus even the two training videos are not two verified equivalent action examples.

First verify visibility and action semantics using training videos; obtain additional independent examples and negatives. Do not turn this catalog tag into a legal multiclass target. The current data cannot support a credible evaluated action classifier.

No second type is selected merely to fill a quota. client_identity_check_not_visible has only one train and one validation video and depends on absence from camera view. safe_key_left_in_lock has two train videos but zero validation videos. The other types have one source example each. A type present only in validation cannot be trained from these data; a type absent from validation cannot be evaluated there. Do not move clips or harvest validation examples for training.

## Applicability and ambiguity by type

These conditions are copied from the existing repository catalog, not newly verified legal interpretations. The mapping does not confirm conditions outside the camera view. Binary full-video inference remains independent of this audit.

### unsecured_accepted_cash

Exploratory only; too few independent videos and unverified action tags.

Applicability from catalog:

- Деньги уже приняты банком, а не являются текущим взносом клиента, который ещё пересчитывается в его присутствии.
- Предметы в кадре идентифицируются как наличные, либо оставлена без присмотра сумка, которую команда считает содержащей ранее принятые наличные (X-15).
- Работа с этими деньгами завершена либо это ранее принятые деньги, которые п. 20 абз. 2 требует хранить в закрывающемся устройстве.

Video cannot establish:

- Факт бухгалтерского приёма и завершения операции, если он не следует из видимого действия.
- Содержимое закрытой сумки, если команда не принимает рабочую гипотезу, что внутри наличные.

### cash_counted_without_client_present

Cannot evaluate held-out action recognition with this split.

Applicability from catalog:

- Это именно приём или пересчёт наличных клиента, а не внутренний пересчёт, сортировка или упаковка (для них см. п. 42).

Video cannot establish:

- Была ли операция уже завершена до ухода клиента.

### cashier_leaves_during_client_service

Cannot evaluate held-out action recognition with this split.

Applicability from catalog:

- В кадре ещё идёт обслуживание клиента.
- Если одновременно на столе остаются ранее принятые наличные, дополнительно применим unsecured_accepted_cash / п. 20 абз. 2.

Video cannot establish:

- Завершён ли клиентский цикл, если клиента уже нет в кадре и операция могла быть закрыта.

### client_identity_check_not_visible

Exploratory only; too few independent videos and unverified action tags.

Applicability from catalog:

- Лицо в кадре является клиентом в смысле Правил №231.
- Формулировка результата должна быть «в кадре не наблюдается визуальная идентификация», а не «идентификация не проводилась».

Video cannot establish:

- Проверку вне кадра, через операциониста или цифровой сервис, который не попал в обзор камеры.

### employee_personal_items_in_cash_unit

Cannot evaluate held-out action recognition with this split.

Applicability from catalog:

- Вещи принадлежат работнику банка, а не клиенту.
- Место находится в кассовом узле.
- Это не лекарства, медизделия или средство связи без фото/видеосъёмки, которые п. 12 прямо допускает.

Video cannot establish:

- Принадлежность вещи, если это нельзя отличить по кадру.

### recount_without_visible_controller

Cannot evaluate held-out action recognition with this split.

Applicability from catalog:

- Операция относится к пересчёту, сортировке или упаковке, а не к приёму или выдаче наличных клиенту (эти случаи п. 42 прямо исключает).

Video cannot establish:

- Наличие контролёра вне кадра.
- Вид операции, если по видео нельзя отличить клиентский приём от внутреннего пересчёта.

### collection_cash_amount_registration_not_observed

Cannot evaluate held-out action recognition with this split.

Applicability from catalog:

- Это приём наличных клиента от инкассации банка или инкассаторской организации по препроводительному документу.
- Команда считает отсутствие видимой регистрации суммы событием для API, хотя сам порядок регистрации задают внутренние документы банка.

Video cannot establish:

- Запись суммы в учётной системе вне кадра.

### cash_bag_received_without_visible_identity_authority_check

Cannot evaluate held-out action recognition with this split.

Applicability from catalog:

- Лицо передаёт наличные как клиент в смысле №231, п. 19 пп. 1, либо как представитель клиента, к которому применима ст. 5 Закона №191-IV.
- Формулировка: «в кадре не наблюдается проверка личности и полномочий», а не «проверка не проводилась».
- Ст. 7 Закона №191-IV задаёт момент надлежащей проверки и отдельно как тип не используется.

Video cannot establish:

- Статус клиента или представителя и то, требовалась ли CDD по сумме и виду операции.

### safe_key_left_in_lock

Cannot evaluate held-out action recognition with this split.

Applicability from catalog:

- П. 80 требует, чтобы внутренние документы банка исключали несанкционированное использование ключей от сейфа, кладовой и устройств хранения.
- Команда трактует ключ, оставленный в замке, как такое несанкционированное использование, видимое на видео.

Video cannot establish:

- Текст внутреннего регламента ForteBank и точный запрет «не оставлять ключ в замке».

### suspected_stamp_or_seal_left_unattended

Cannot evaluate held-out action recognition with this split.

Applicability from catalog:

- П. 80 относит хранение печатей, клише и именных штампов к внутренним документам, которые должны исключать несанкционированное использование.
- Команда считает предметы в кадре банковскими печатями или штампами.

Video cannot establish:

- Точная идентификация предмета, если по кадру нельзя отличить печать банка от иного предмета.

## Unmapped positives

X-1, X-28, X-30, X-32, X-33, X-46. Their accepted binary labels are unchanged; no supported action type is invented.

Reproduce with rules.coverage.catalog_coverage(events_path, split_path) or notebook 02 section 10. No split, annotation, historical checkpoint or metric is rewritten.
