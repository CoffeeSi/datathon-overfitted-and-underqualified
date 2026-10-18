# Video labels: snapshot of events.csv

Readable snapshot of `data/annotations/events.csv`: **42 records**,
**22 violation** and **20 no_violation**. The CSV is the source; edit it first
and regenerate this snapshot. The team accepts its binary labels and single event
timestamps for the current baseline; no separate approval table is required.

Source labels and time markers provide the supplied supervision and its mechanical
conversions. Descriptions, event tags, candidate categories and rule references
are **preliminary interpretations of the supplied text**, not independently
verified observations or labels produced by video recognition. Missing negative-row
descriptions were subsequently filled from sampled video review; see
[provenance](../reports/visible_descriptions_20261018.md). They are not ready
multiclass training targets. No review-completion status or confidence is stored.

All current columns are copied in CSV order without rewording. `—` displays an
empty CSV cell. See [video-label-schema.md](video-label-schema.md) for field
interpretation and intended use. "Split" in this filename refers to event/category
separation; training/validation assignments live in `data/processed/split_proposal.csv`.

| `﻿clip_id` | `timecode` | `source_label_ru` | `observed_event` | `observed_event_type` | `candidate_violation_type` | `legal_basis` | `source_label` | `timecode_sec` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X-1 | 00:22 | Нарушение | Кассир получила наличные, пересчитала их на машинке и вернула часть купюр клиенту | cash_received_counted_partially_returned | cash_receipt_accounting | Постановление НБРК №231, п. 19 пп. 4 | violation | 22.0 |
| X-2 | 00:10 | Нарушение | В кадре не наблюдается визуальная идентификация клиента, вносящего наличные | cash_deposit_without_visible_id_check | customer_identification | №231, п. 19, пп. 1 | violation | 10.0 |
| X-3 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-4 | 00:57 | Нарушение | Деньги клиента №1 остались на столе при входе клиента №2 | prior_client_cash_left_during_next_client_entry | cash_storage | №231, п. 20, абз. 2 | violation | 57.0 |
| X-5 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-8 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-10 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-11 | 00:18 | Нарушение | Ключ оставлен в замке сейфа | safe_key_left_in_lock | key_storage | №231, п. 80 | violation | 18.0 |
| X-12 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-13 | 00:47 | Нарушение | В видео не наблюдается регистрация суммы денег из инкассаторской сумки до помещения на хранение | collection_cash_amount_registration_not_observed | collection_cash_accounting | — | violation | 47.0 |
| X-14 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-15 | 00:15 | Нарушение | Сумка оставлена без присмотра; содержимое не подтверждено | unattended_bag_contents_unknown | cash_storage | — | violation | 15.0 |
| X-16 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-17 | 00:07 | Нарушение | Сумка с наличными принята; проверка личности и полномочий передавшего лица в кадре не наблюдается | cash_bag_received_without_visible_identity_authority_check | identity_and_authority_check | №231, п. 19, пп. 1 | violation | 7.0 |
| X-18 | 00:20 | Нарушение | В кадре не наблюдается визуальная идентификация клиента, вносящего наличные | cash_deposit_without_visible_id_check | customer_identification | №231, п. 19, пп. 1 | violation | 20.0 |
| X-19 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-20 | 00:05 | Нарушение | Сейф открыт; кассир пересчитывает деньги из сейфа; контролирующий работник в кадре не наблюдается | cash_count_from_safe_without_visible_controller | cash_storage_or_recount_control | — | violation | 5.0 |
| X-21 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-22 | 00:14 | Нарушение | Сотрудница отходит, ключ остаётся в замке сейфа | employee_steps_away_key_left_in_safe_lock | key_storage | №231, п. 80 | violation | 14.0 |
| X-23 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-24 | 00:08 | Нарушение | Предполагаемые банковские печати или штампы оставлены на столе без присмотра | suspected_stamp_or_seal_left_unattended | stamp_and_seal_storage | №231, п. 80 | violation | 8.0 |
| X-25 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-26 | 00:09 | Нарушение | Сотрудница отходит, предполагаемые наличные остаются на рабочем столе | employee_steps_away_suspected_cash_on_desk | cash_storage_and_workstation_abandonment | №231, п. 20, абз. 2; п. 22 | violation | 9.0 |
| X-27 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-28 | 00:07 | Нарушение | Сотрудница отходит от ноутбука; состояние блокировки рабочей сессии не установлено | employee_steps_away_workstation_lock_unknown | information_security | — | violation | 7.0 |
| X-29 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-30 | 00:13 | Нарушение | Мужчина входит в рабочую кассовую зону; его полномочия не установлены | person_enters_cash_work_area_authorization_unknown | access_control | — | violation | 13.0 |
| X-31 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-32 | 00:05 | Нарушение | Сотрудница работает возле кассового оборудования при открытой двери | cash_area_door_open_during_work | cash_area_access_control | — | violation | 5.0 |
| X-33 | 00:06 | Нарушение | Сотрудница садится за рабочее место, входная дверь остаётся открытой | employee_starts_work_entry_door_open | cash_area_access_control | — | violation | 6.0 |
| X-34 | 00:13 | Нарушение | Кассир пересчитывает принимаемые наличные после ухода клиента | cash_count_after_client_departure | cash_acceptance_without_client_present | №231, п. 20, абз. 1 | violation | 13.0 |
| X-35 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-37 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-38 | 00:33 | Нарушение | Клиент уходит; предполагаемые принятые деньги остаются на столе | suspected_accepted_cash_left_after_client_departure | cash_storage | №231, п. 20, абз. 2 | violation | 33.0 |
| X-39 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-40 | 00:17 | Нарушение | Личная сумка и верхняя одежда размещены на кассовом столе | employee_personal_items_on_cashier_desk | personal_items_in_cash_unit | №231, п. 12 | violation | 17.0 |
| X-43 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-44 | 00:16 | Нарушение | Сотрудница работает с предполагаемыми банкнотами без клиента | employee_handles_suspected_cash_without_client | cash_storage | №231, п. 20, абз. 2 | violation | 16.0 |
| X-45 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| X-46 | 00:11 | Нарушение | На рабочем месте сотрудницы находится напиток рядом с предполагаемыми денежными средствами | beverage_near_suspected_cash | workplace_organization | — | violation | 11.0 |
| X-47 | 01:08 | Нарушение | После приёма наличные не помещены в закрывающееся устройство хранения | accepted_cash_not_moved_to_lockable_storage | cash_storage | №231, п. 20, абз. 2 | violation | 68.0 |
| X-48 | — | Без нарушения | Событие отдельно не описано в исходной разметке | not_reported | — | — | no_violation | — |
| `clip_id` | `timecode` | `source_label_ru` | `observed_event` | `observed_event_type` | `candidate_violation_type` | `legal_basis` | `source_label` | `timecode_sec` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X-1 | 00:22 | Нарушение | Кассир получила наличные, пересчитала их на машинке и вернула часть купюр клиенту | cash_received_counted_partially_returned | cash_receipt_accounting | Постановление НБРК №231, п. 19 пп. 4 | violation | 22.0 |
| X-2 | 00:10 | Нарушение | В кадре не наблюдается визуальная идентификация клиента, вносящего наличные | cash_deposit_without_visible_id_check | customer_identification | Постановление НБРК №231, п. 19, пп. 1 | violation | 10.0 |
| X-3 | — | Без нарушения | Посетитель подходит к столу и садится напротив кассира; кассир рассматривает переданный небольшой документ, затем работает с предметами на столе и кассовым оборудованием. | not_reported | — | — | no_violation | — |
| X-4 | 00:57 | Нарушение | Деньги клиента №1 остались на столе при входе клиента №2 | prior_client_cash_left_during_next_client_entry | cash_storage | Постановление НБРК №231, п. 20, ч. 2 | violation | 57.0 |
| X-5 | — | Без нарушения | Кассир обслуживает посетителя: рассматривает переданный документ, работает за компьютером и с кассовым оборудованием; посетитель уходит, после чего к столу подходит и садится следующий. | not_reported | — | — | no_violation | — |
| X-8 | — | Без нарушения | Кассир пересчитывает банкноты вручную, обращается к металлическому сейфу справа от стола, затем достаёт и раскладывает на столе листы бумаги. | not_reported | — | — | no_violation | — |
| X-10 | — | Без нарушения | Кассир перебирает и раскладывает банкноты на столе, обращается к счётной машинке и перекладывает пачки купюр. | not_reported | — | — | no_violation | — |
| X-11 | 00:18 | Нарушение | Ключ оставлен в замке сейфа | safe_key_left_in_lock | key_storage | Постановление НБРК №231, п. 80 | violation | 18.0 |
| X-12 | — | Без нарушения | Кассир работает за компьютером, затем встаёт, обращается к металлическому сейфу за рабочим местом, выключает свет и выходит из кабинета. | not_reported | — | — | no_violation | — |
| X-13 | 00:47 | Нарушение | В видео не наблюдается регистрация суммы денег из инкассаторской сумки до помещения на хранение | collection_cash_amount_registration_not_observed | collection_cash_accounting | Постановление НБРК №231, п. 24, пп. 2 | violation | 47.0 |
| X-14 | — | Без нарушения | Сотрудница открывает зелёную сумку на столе, достаёт и раскладывает банкноты, убирает сумку, затем работает с купюрами, счётной машинкой, бумагами и компьютером. | not_reported | — | — | no_violation | — |
| X-15 | 00:15 | Нарушение | Сумка оставлена без присмотра; содержимое не подтверждено | unattended_bag_contents_unknown | cash_storage | Постановление НБРК №231, п. 20, ч. 2 | violation | 15.0 |
| X-16 | — | Без нарушения | Посетитель приносит зелёную сумку и ставит её на стол, передаёт небольшой документ; кассир рассматривает документ, после чего посетитель открывает сумку и садится напротив. | not_reported | — | — | no_violation | — |
| X-17 | 00:07 | Нарушение | Сумка с наличными принята; проверка личности и полномочий передавшего лица в кадре не наблюдается | cash_bag_received_without_visible_identity_authority_check | identity_and_authority_check | Постановление НБРК №231, п. 19, пп. 1 | violation | 7.0 |
| X-18 | 00:20 | Нарушение | В кадре не наблюдается визуальная идентификация клиента, вносящего наличные | cash_deposit_without_visible_id_check | customer_identification | Постановление НБРК №231, п. 19, пп. 1 | violation | 20.0 |
| X-19 | — | Без нарушения | Кассир рассматривает переданный посетителем документ, работает с кассовым оборудованием и обращается к металлическому сейфу справа; посетитель уходит, затем приходит следующий. | not_reported | — | — | no_violation | — |
| X-20 | 00:05 | Нарушение | Сейф открыт; кассир пересчитывает деньги из сейфа; контролирующий работник в кадре не наблюдается | cash_count_from_safe_without_visible_controller | cash_storage_or_recount_control | Постановление НБРК №231, п. 42; пункт 20, абзац 2 | violation | 5.0 |
| X-21 | — | Без нарушения | Сотрудница достаёт пачку банкнот из металлического сейфа справа от стола, садится на рабочее место и помещает купюры в счётную машинку. | not_reported | — | — | no_violation | — |
| X-22 | 00:14 | Нарушение | Сотрудница отходит, ключ остаётся в замке сейфа | employee_steps_away_key_left_in_safe_lock | key_storage | Постановление НБРК №231, п. 80 | violation | 14.0 |
| X-23 | — | Без нарушения | Сотрудница работает за компьютером, затем встаёт, обращается к металлическому сейфу справа от стола и выходит из кабинета. | not_reported | — | — | no_violation | — |
| X-24 | 00:08 | Нарушение | Предполагаемые банковские печати или штампы оставлены на столе без присмотра | suspected_stamp_or_seal_left_unattended | stamp_and_seal_storage | Постановление НБРК №231, п. 80 | violation | 8.0 |
| X-25 | — | Без нарушения | Кассир работает с бумагами и небольшими предметами на столе, затем переносит часть предметов в металлический сейф справа, закрывает его дверцу и выходит из кабинета. | not_reported | — | — | no_violation | — |
| X-26 | 00:09 | Нарушение | Сотрудница отходит, предполагаемые наличные остаются на рабочем столе | employee_steps_away_suspected_cash_on_desk | cash_storage_and_workstation_abandonment | Постановление НБРК №231, п. 20, ч. 2 | violation | 9.0 |
| X-27 | — | Без нарушения | Кассир работает с бумагами за столом, затем встаёт, обращается к металлическому сейфу справа, закрывает его дверцу и выходит из кабинета. | not_reported | — | — | no_violation | — |
| X-28 | 00:07 | Нарушение | Сотрудница отходит от ноутбука; состояние блокировки рабочей сессии не установлено | employee_steps_away_workstation_lock_unknown | information_security | Постановление АРРФР №53, п. 53 | violation | 7.0 |
| X-29 | — | Без нарушения | Сотрудница работает за компьютером, затем встаёт со стула и выходит из кабинета; рабочее место остаётся пустым. | not_reported | — | — | no_violation | — |
| X-30 | 00:13 | Нарушение | Мужчина входит в рабочую кассовую зону; его полномочия не установлены | person_enters_cash_work_area_authorization_unknown | access_control | Постановление НБРК №14, п. 8 | violation | 13.0 |
| X-31 | — | Без нарушения | Сотрудница входит в кабинет, закрывает за собой стеклянную дверь, садится к компьютеру, затем встаёт и обращается к нижней части рабочего стола. | not_reported | — | — | no_violation | — |
| X-32 | 00:05 | Нарушение | Сотрудница работает возле кассового оборудования при открытой двери | cash_area_door_open_during_work | cash_area_access_control | Постановление НБРК №14, п. 8 | violation | 5.0 |
| X-33 | 00:06 | Нарушение | Сотрудница садится за рабочее место, входная дверь остаётся открытой | employee_starts_work_entry_door_open | cash_area_access_control | Постановление НБРК №14, п. 8 | violation | 6.0 |
| X-34 | 00:13 | Нарушение | Кассир пересчитывает принимаемые наличные после ухода клиента | cash_count_after_client_departure | cash_acceptance_without_client_present | Постановление НБРК №231, п. 20, ч. 1 | violation | 13.0 |
| X-35 | — | Без нарушения | Посетитель садится напротив кассира; кассир работает с банкнотами и счётной машинкой, затем открывает и закрывает металлический сейф справа от стола; посетитель уходит. | not_reported | — | — | no_violation | — |
| X-37 | — | Без нарушения | Посетитель входит и садится напротив кассира; кассир берёт банкноты со стола и помещает их в счётную машинку. | not_reported | — | — | no_violation | — |
| X-38 | 00:33 | Нарушение | Клиент уходит; предполагаемые принятые деньги остаются на столе | suspected_accepted_cash_left_after_client_departure | cash_storage | Постановление НБРК №231, п. 20, абз. 2 | violation | 33.0 |
| X-39 | — | Без нарушения | Кассир последовательно обслуживает двух посетителей: пересчитывает банкноты с помощью машинки, открывает и закрывает металлический сейф справа от стола; посетители по очереди уходят. | not_reported | — | — | no_violation | — |
| X-40 | 00:17 | Нарушение | Личная сумка и верхняя одежда размещены на кассовом столе | employee_personal_items_on_cashier_desk | personal_items_in_cash_unit | Постановление НБРК №231, п. 12 | violation | 17.0 |
| X-43 | — | Без нарушения | Кассир пересчитывает банкноты вручную и на счётной машинке, после чего кладёт пачку купюр на стол. | not_reported | — | — | no_violation | — |
| X-44 | 00:16 | Нарушение | Сотрудница работает с предполагаемыми банкнотами без клиента | employee_handles_suspected_cash_without_client | cash_storage | Постановление НБРК №231, п. 20, абз. 2 | violation | 16.0 |
| X-45 | — | Без нарушения | Кассир пересчитывает банкноты вручную и с помощью счётной машинки, раскладывает и перекладывает пачки на рабочей поверхности. | not_reported | — | — | no_violation | — |
| X-46 | 00:11 | Нарушение | На рабочем месте сотрудницы находится напиток рядом с предполагаемыми денежными средствами | beverage_near_suspected_cash | workplace_organization | Постановление НБРК №231, п. 12 | violation | 11.0 |
| X-47 | 01:08 | Нарушение | После приёма наличные не помещены в закрывающееся устройство хранения | accepted_cash_not_moved_to_lockable_storage | cash_storage | Постановление НБРК №231, п. 20, абз. 2 | violation | 68.0 |
| X-48 | — | Без нарушения | Кассир обслуживает посетителя: рассматривает переданный документ, работает за компьютером и с кассовым оборудованием; посетитель уходит, после чего к столу подходит и садится следующий. | not_reported | — | — | no_violation | — |
