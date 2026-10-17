# Video labels: event types and candidate violation types

This is the supplied 42-clip classification reorganized into separate fields:

- `observed_event_type` names the event described from the video. It should stay descriptive and preserve uncertainty.
- `candidate_violation_type` names the possible compliance category. It is a review hypothesis, not a final legal finding.
- `source_label` preserves the original dataset label. A `Без нарушения` label means the supplied annotation reported no violation; it does not mean every possible event was ruled out.

For clips marked `Без нарушения`, the source does not describe a separate event, so `observed_event_type` is `not_reported` and the candidate violation type is blank. This avoids treating the negative source label as a claim that nothing happened in the video.

| File | Timestamp | Source label | Observed event | `observed_event_type` | `candidate_violation_type` | Candidate rule / qualification |
| --- | --- | --- | --- | --- | --- | --- |
| X-1 | 00:22 | Нарушение | После завершения пересчёта принятые деньги оставлены на рабочем столе | `accepted_cash_left_on_desk_after_count` | `cash_storage` | №231, п. 20, абз. 2 |
| X-2 | 00:10 | Нарушение | В кадре не наблюдается визуальная идентификация клиента, вносящего наличные | `cash_deposit_without_visible_id_check` | `customer_identification` | №231, п. 19, пп. 1, если это клиент в смысле Правил |
| X-3 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-4 | 00:57 | Нарушение | Деньги клиента №1 остались на столе при входе клиента №2 | `prior_client_cash_left_during_next_client_entry` | `cash_storage` | №231, п. 20, абз. 2; подтвердить, что деньги уже приняты |
| X-5 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-8 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-10 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-11 | 00:18 | Нарушение | Ключ оставлен в замке сейфа | `safe_key_left_in_lock` | `key_storage` | №231, п. 80 — отсылка к внутреннему регламенту банка |
| X-12 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-13 | 00:47 | Нарушение | В видео не наблюдается регистрация суммы денег из инкассаторской сумки до помещения на хранение | `collection_cash_amount_registration_not_observed` | `collection_cash_accounting` | №231, п. 24, пп. 2 — подтвердить по внутреннему порядку и учётным записям |
| X-14 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-15 | 00:15 | Нарушение | Сумка оставлена без присмотра; содержимое не подтверждено | `unattended_bag_contents_unknown` | `cash_storage` | №231, п. 20, абз. 2 — только если внутри ранее принятые клиентские деньги |
| X-16 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-17 | 00:07 | Нарушение | Сумка с наличными принята; проверка личности и полномочий передавшего лица в кадре не наблюдается | `cash_bag_received_without_visible_identity_authority_check` | `identity_and_authority_check` | №231, п. 19, пп. 1; Закон №191-IV, ст. 5, п. 3, пп. 1 и 6; ст. 7, п. 1 — если лицо действует как представитель клиента и операция подпадает под требования закона |
| X-18 | 00:20 | Нарушение | В кадре не наблюдается визуальная идентификация клиента, вносящего наличные | `cash_deposit_without_visible_id_check` | `customer_identification` | №231, п. 19, пп. 1, если это клиент в смысле Правил |
| X-19 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-20 | 00:05 | Нарушение | Сейф открыт; кассир пересчитывает деньги из сейфа; контролирующий работник в кадре не наблюдается | `cash_count_from_safe_without_visible_controller` | `cash_storage_or_recount_control` | №231, п. 20, абз. 2; п. 42; п. 80 — разные условия применения; подтвердить вид операции и отсутствие контролёра |
| X-21 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-22 | 00:14 | Нарушение | Сотрудница отходит, ключ остаётся в замке сейфа | `employee_steps_away_key_left_in_safe_lock` | `key_storage` | №231, п. 80 — отсылка к внутреннему регламенту банка |
| X-23 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-24 | 00:08 | Нарушение | Предполагаемые банковские печати или штампы оставлены на столе без присмотра | `suspected_stamp_or_seal_left_unattended` | `stamp_and_seal_storage` | №231, п. 80 — отсылка к внутреннему регламенту; подтвердить, что это банковские печати или штампы |
| X-25 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-26 | 00:09 | Нарушение | Сотрудница отходит, предполагаемые наличные остаются на рабочем столе | `employee_steps_away_suspected_cash_on_desk` | `cash_storage_and_workstation_abandonment` | №231, п. 20, абз. 2; п. 22 — если обслуживание клиента ещё не завершено |
| X-27 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-28 | 00:07 | Нарушение | Сотрудница отходит от ноутбука; состояние блокировки рабочей сессии не установлено | `employee_steps_away_workstation_lock_unknown` | `information_security` | Внутренние правила ИБ ForteBank; точный пункт не установлен |
| X-29 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-30 | 00:13 | Нарушение | Мужчина входит в рабочую кассовую зону; его полномочия не установлены | `person_enters_cash_work_area_authorization_unknown` | `access_control` | Внутренний пропускной режим; Правила №14, п. 8 относятся к изоляции кассового узла, а не подтверждают отсутствие полномочий |
| X-31 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-32 | 00:05 | Нарушение | Сотрудница работает возле кассового оборудования при открытой двери | `cash_area_door_open_during_work` | `cash_area_access_control` | Правила №14, п. 8; внутренний регламент — открытая дверь сама по себе не доказывает нарушение |
| X-33 | 00:06 | Нарушение | Сотрудница садится за рабочее место, входная дверь остаётся открытой | `employee_starts_work_entry_door_open` | `cash_area_access_control` | Правила №14, п. 8; внутренний регламент — уточнить, какие помещения соединяет дверь |
| X-34 | 00:13 | Нарушение | Кассир пересчитывает принимаемые наличные после ухода клиента | `cash_count_after_client_departure` | `cash_acceptance_without_client_present` | №231, п. 20, абз. 1 — если продолжается приём и пересчёт наличных клиента |
| X-35 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-37 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-38 | 00:33 | Нарушение | Клиент уходит; предполагаемые принятые деньги остаются на столе | `suspected_accepted_cash_left_after_client_departure` | `cash_storage` | №231, п. 20, абз. 2 — подтвердить приём денег и завершение операции |
| X-39 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-40 | 00:17 | Нарушение | Личная сумка и верхняя одежда размещены на кассовом столе | `employee_personal_items_on_cashier_desk` | `personal_items_in_cash_unit` | №231, п. 12 — если это вещи сотрудника и место находится в кассовом узле |
| X-43 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-44 | 00:16 | Нарушение | Сотрудница работает с предполагаемыми банкнотами без клиента | `employee_handles_suspected_cash_without_client` | `cash_storage` | №231, п. 20, абз. 2 — только если это ранее принятые деньги и работа с ними завершена |
| X-45 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |
| X-46 | 00:11 | Нарушение | На рабочем месте сотрудницы находится напиток рядом с предполагаемыми денежными средствами | `beverage_near_suspected_cash` | `workplace_organization` | Внутренний регламент ForteBank; №231, п. 12 — только условная ссылка |
| X-47 | 01:08 | Нарушение | После приёма наличные не помещены в закрывающееся устройство хранения | `accepted_cash_not_moved_to_lockable_storage` | `cash_storage` | №231, п. 20, абз. 2 |
| X-48 | — | Без нарушения | Событие отдельно не описано в исходной разметке | `not_reported` | — | — |

## Tag interpretation

`observed_event_type` should describe a visible action or state. It must not imply facts that the clip cannot establish, such as whether money was formally accepted, a client check happened off-camera, a laptop session was unlocked, or an entering person lacked authorization.

`candidate_violation_type` groups events by the possible compliance issue. It may be blank for a source `Без нарушения` row, and it may remain conditional until transaction records, internal procedures, room layout, or other evidence is checked. The source label is preserved independently of both type fields.

The legal references are candidate mappings, not definitive legal conclusions. See [video-label-schema.md](video-label-schema.md) for the annotation protocol and legal review caveats.
