# Video Label Schema and Review Notes

**Status:** Provisional reference for the supplied 42-clip classification. The original dataset labels are preserved; the event descriptions and legal mappings are review annotations, not final findings of legal or employee misconduct.

## Purpose

This file records what each label means and how to keep three things separate:

1. **Source label:** the binary label supplied with the dataset (`Нарушение` / `Без нарушения`).
2. **Observed event:** a description of what the video visibly shows, with a timestamp and uncertainty where needed.
3. **Review interpretation:** a possible violation category and a candidate legal or internal-policy reference, subject to factual and applicability checks.

Do not change the source label just because a legal mapping is uncertain. Conversely, a source label of `Без нарушения` means that the supplied annotation found no violation; it does not prove that no violation occurred.

## Fields for future annotation records

| Field | Meaning |
| --- | --- |
| `clip_id` | Exact source filename/ID. Preserve the original spelling, including visually similar Latin and Cyrillic characters. |
| `source_label` | `violation` or `no_violation`, copied from the supplied dataset. |
| `timecode` | Event time in the source clip; null/blank when the supplied label has no timestamp. |
| `source_event_ru` | Original Russian event wording from the supplied table. Keep it unchanged for traceability. |
| `observed_event` | Plain, visual description. Use wording such as “предполагаемые наличные” where the object or its status is uncertain. |
| `observed_event_type` | A concise, consistent tag for the visible event, with uncertainty preserved. |
| `candidate_violation_type` | A separate tag for the possible compliance category. It is a review hypothesis, not a legal conclusion. |
| `legal_basis` | Candidate provision or internal policy, plus any condition that must be established. |
| `review_status` | `basis_identified`, `conditional`, `internal_policy_required`, or `not_mapped`. These describe review readiness, not whether misconduct is proven. |
| `evidence_note` | What is visible, what is not visible, and what additional evidence is needed. |

If object detection is later needed, add `actor_id`, `object_id`, visibility/occlusion, and per-frame boxes/tracks as separate annotation fields. Do not encode legal conclusions as object or action labels.

The full 42-row table with `observed_event_type` and `candidate_violation_type` split into separate fields is in [video-labels-split.md](video-labels-split.md).

## Candidate event catalog

The canonical tags below normalize the wording in the supplied table. Keep the Russian observation alongside the tag when exporting or reviewing labels.

| Clip | Time | Candidate tag | Original event wording (RU) and qualification from the supplied classification |
| --- | --- | --- | --- |
| X-1 | 00:22 | `cash.unsecured_after_acceptance` | «После завершения пересчёта принятые деньги оставлены на рабочем столе». Candidate basis: №231, p. 20, para. 2. |
| X-2 | 00:10 | `client.identity_check_not_observed` | «Клиент внёс наличные без визуальной идентификации». №231, p. 19(1), if the person is a client within the Rules. |
| X-4 | 00:57 | `cash.left_between_client_transactions` | «Деньги клиента №1 остались на столе при входе клиента №2». Candidate basis: №231, p. 20, para. 2; confirm that the cash was already accepted. |
| X-11 | 00:18 | `key.left_in_safe_lock` | «Ключ оставлен в замке сейфа». №231, p. 80 refers the storage procedure to bank internal documents. |
| X-13 | 00:47 | `collection.cash_amount_not_registered` | «Деньги из инкассаторской сумки помещены на хранение без регистрации суммы». №231, p. 24(2); applicability needs review. |
| X-15 | 00:15 | `bag_unattended_contents_uncertain` | «Сумка с предполагаемыми наличными оставлена без присмотра». №231, p. 20, para. 2 applies only if it contains previously accepted client cash. |
| X-17 | 00:07 | `bag_received_without_visible_identity_or_authority_check` | «Сумка с наличными принята без проверки личности и полномочий передавшего лица». Candidate references: №231, p. 19(1); Law №191-IV, art. 5(3)(1), (6), and art. 7(1). The AML-law references apply only if the person acts for a client, the operation is within the law's scope, and due-diligence measures are required. Art. 7(1) sets the timing for those measures; it is not a separate general identity-check rule. |
| X-18 | 00:20 | `client.identity_check_not_observed` | «Не проведена визуальная идентификация клиента, вносящего наличные». №231, p. 19(1), if the person is a client within the Rules. |
| X-20 | 00:05 | `safe_open_cash_count_without_controller` | «Сейф оставлен открытым; кассир самостоятельно пересчитывает деньги из сейфа». The clip-to-event attribution remains provisional; p. 20(2), p. 42, and p. 80 have different conditions and should not be treated as interchangeable. |
| X-22 | 00:14 | `key_left_in_safe_lock_employee_away` | «Сотрудница отошла, оставив ключ в замке сейфа». №231, p. 80; check the bank's internal procedure. |
| X-24 | 00:08 | `stamps_or_seals_left_unattended` | «Предполагаемые банковские печати или штампы оставлены на столе без присмотра». №231, p. 80 refers the procedure to internal documents; confirm the items' identity. |
| X-26 | 00:09 | `cash_left_out_when_employee_leaves` | «Сотрудница отошла, оставив предполагаемые наличные на рабочем столе». №231, p. 20(2) if it is previously accepted cash; p. 22 if client service is still in progress. |
| X-28 | 00:07 | `workstation_lock_state_unconfirmed` | «Сотрудница отходит от ноутбука; возможна незаблокированная рабочая сессия». Whether the session is unlocked is not established by the description. Internal information-security rule and exact clause are needed. |
| X-30 | 00:13 | `person_enters_cash_area_authorization_unknown` | «Мужчина входит в рабочую кассовую зону». №14, p. 8 concerns isolation of the cash unit; authorization must be checked under the internal access regime. |
| X-32 | 00:05 | `cash_area_door_open` | «Сотрудница работает возле кассового оборудования при открытой двери». №14, p. 8 and internal rules are candidates; an open door in a frame alone does not establish that the cash unit fails the isolation requirement. |
| X-33 | 00:06 | `employee_starts_work_door_open` | «Сотрудница садится за рабочее место, не закрыв входную дверь». Same limitation as X-32: confirm which door it is, what rooms it connects, and the applicable internal procedure. |
| X-34 | 00:13 | `cash_count_after_client_departure` | «Кассир пересчитывает принимаемые наличные после ухода клиента». №231, p. 20(1) if this is still the acceptance and recount of that client's cash. |
| X-38 | 00:33 | `suspected_accepted_cash_left_after_client_departure` | «Клиент ушёл, предполагаемые принятые деньги остались на столе». №231, p. 20(2) if acceptance and completion of the operation are confirmed. |
| X-40 | 00:17 | `employee_personal_items_at_cash_desk` | «Личная сумка и верхняя одежда размещены на кассовом столе». №231, p. 12 if these are employee personal belongings and the location is inside the cash unit. |
| X-44 | 00:16 | `cash_handling_without_client_acceptance_status_unknown` | «Сотрудница работает с предполагаемыми банкнотами без клиента». №231, p. 20(2) only after work with the cash is complete and the cash is previously accepted. |
| X-46 | 00:11 | `beverage_near_suspected_cash` | «На рабочем месте сотрудницы находится напиток рядом с предполагаемыми денежными средствами». Bank internal workplace rules are needed; p. 12 is only a conditional reference. |
| X-47 | 01:08 | `cash.not_moved_to_lockable_storage` | «После приёма наличных сотрудница не убрала их в закрывающееся устройство хранения». Candidate basis: №231, p. 20(2); confirm the acceptance and device context. |

## Supplied `no_violation` rows

The source classification has no violation timestamp for these 20 clips: **X-3, X-5, X-8, X-10, X-12, X-14, X-16, X-19, X-21, X-23, X-25, X-27, X-29, X-31, X-35, X-37, X-39, X-43, X-45, X-48**. Their supplied event wording is «По разметке нарушение не выявлено». Preserve them as `no_violation` in the source-label field. Do not infer that the clips were independently reviewed for every category in the event catalog.

## Review-status guide

- **`basis_identified`** — a plausible provision is identified, but the video facts still need confirmation. Examples noted in the review: X-1, X-4, X-40, X-47.
- **`conditional`** — a relevant provision may apply only after confirming a fact such as client status, representation, contents of a bag, acceptance of cash, operation type, or transaction completion. Examples include X-2, X-13, X-15, X-17, X-18, X-20, X-26, X-34, X-38, and X-44.
- **`internal_policy_required`** — the cited public rule delegates the procedure or the issue depends on bank policy/access controls. Examples include X-11, X-22, X-24, X-28, X-30, X-32, X-33, and X-46.
- **`not_mapped`** — no candidate event or source is assigned. This applies to supplied negative rows unless a separate review records otherwise.

These examples are triage notes from the supplied classification, not a complete legal assessment. Some clips may need more than one status because evidence certainty and source applicability are separate questions.

## Annotation rules

1. Record what is visible before assigning a candidate type. Prefer “door is open” to “unauthorized access”; prefer “object resembling banknotes” to “cash” when the image is unclear.
2. Preserve uncertainty explicitly: `предполагаемые`, `возможно`, `не видно`, or `не установлено`. Lack of visible evidence is not proof that an action did not happen.
3. Use the timestamp as the start of the visible event. If duration matters, add an end time or interval rather than treating a single timestamp as the whole event.
4. Keep legal mapping separate. A provision may depend on transaction status, client/representative status, employee ownership, room layout, internal policy, or other evidence unavailable from the clip.
5. Treat `violation` and `no_violation` as source-dataset labels. Any product output should be framed as a review finding, with evidence and uncertainty, not a definitive legal or disciplinary conclusion.
6. Preserve exact filenames when joining annotations to videos. Some IDs may contain visually similar Latin and Cyrillic characters.
7. For X-2 and X-18, describe only what the video establishes: use “в кадре не наблюдается визуальная сверка с документом” unless the full interaction confirms that no check occurred. For X-17, the clip alone may not establish whether the person was a client representative or whether the applicable AML checks were required.
8. For X-30, Rule №14, p. 8 concerns physical isolation of the cash unit; it does not establish that a person entering the area lacked authorization. For X-32 and X-33, an open door alone does not establish failure of that isolation requirement. Use the bank's access and door-control procedures to assess those events.
9. For X-13, the video may not establish whether the receipt amount was registered in the bank's system or records. Confirm against the applicable internal procedure and transaction records before treating “без регистрации суммы” as a verified fact.

## Normative references cited in the supplied classification

- [National Bank Rules №231 — cash operations and collection](https://adilet.zan.kz/rus/docs/V1900019680): cited provisions include p. 12, p. 19(1), p. 20(1)–(2), p. 22, p. 24(2), p. 42, and p. 80.
- [Rules №14 — protection and layout of bank premises](https://adilet.zan.kz/rus/docs/V2000020075): p. 8 is a premises-isolation requirement. It does not by itself establish that a person seen in the cash area lacked authorization or that a door seen open proves noncompliance.
- [Law №191-IV](https://adilet.zan.kz/rus/docs/Z090000191_): cited for X-17 only under the stated representative/client conditions.

Recheck the current official text and the bank's applicable internal policies before using any mapping in a final compliance or disciplinary decision.
