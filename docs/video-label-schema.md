# Video annotation schema

This document describes the current `data/annotations/events.csv` consumed by
`notebooks/01_eda.ipynb`. The CSV is the source of annotation records; this document
is the field contract. [video-labels-split.md](video-labels-split.md) is a readable
snapshot of that CSV, not an independently maintained annotation source.

## Current CSV fields

Fields below follow the current CSV column order. Empty cells mean missing or
not assigned; they do not mean that an event was independently ruled out.

- `clip_id`: normalized, unique video identifier, such as `X-1`. Latin `X` and
  Cyrillic `Х` in video filenames map to Latin `X`; numeric leading zeros are
  removed. Original filenames and paths remain in the video inventory and the
  enriched output, not in this identifier.
- `timecode`: source event marker, written as `MM:SS` in the current CSV. It is
  empty for `no_violation`. The marker is not a verified event start or end.
  EDA also accepts nonnegative seconds or `HH:MM:SS`; subordinate minute/second
  components must be below 60. A violation needs a valid marker inside its video.
- `source_label_ru`: Russian source label: `Нарушение` or `Без нарушения`.
- `observed_event`: current Russian description of the reported visible action
  or state, including uncertainty. These descriptions are review interpretations,
  not guaranteed verbatim copies of the original supplied table and not proof of
  facts that cannot be established from video.
- `observed_event_type`: descriptive event tag. Use the exact tags in the current
  CSV; there is no second canonical tag catalog. Different tags can describe
  related actions. For negative rows the current value is `not_reported`.
- `candidate_violation_type`: possible compliance category, separate from the
  event tag. It is a hypothesis, not an additional confirmed training label.
  The current negative rows leave this field empty.
- `legal_basis`: candidate rule or internal-policy reference, including conditions
  and limitations. EDA preserves this text; it does not verify its applicability.
  Current negative rows leave it empty.
- `review_status`: readiness of the interpretation, using one of the statuses
  defined below. This is not a manual-review completion flag.
- `source_label`: machine-readable source label. `violation` corresponds to
  `Нарушение`; `no_violation` corresponds to `Без нарушения`. Both representations
  must agree. Uncertainty about a description or a rule does not automatically
  change the source label.
- `timecode_sec`: numeric seconds corresponding to `timecode`, empty when the
  source marker is absent. EDA recomputes this derived value in memory and in
  `events_enriched.csv`; `timecode` is the authoritative editable marker.
- `confidence`: qualitative confidence in the interpretation, not a model score
  and not evidence that someone completed a video review. Allowed nonempty values
  are `high`, `medium`, and `low`; current records use `medium`, `low`, or blank.
  Current negative rows leave this field empty.

`manual_observed_action`, `manual_confidence`, `source_event_ru`, and
`evidence_note` are not part of the current schema. There are no event intervals,
object boxes, actor tracks, camera groups, or train/validation assignments in
`events.csv`.

## Review statuses

- `basis_identified`: a possible rule has been identified; the video facts and
  applicability still need confirmation.
- `conditional`: applicability depends on facts such as client status, bag
  contents, acceptance of cash, or completion of the operation.
- `internal_policy_required`: an internal bank procedure or access/security
  policy is needed to assess the event.
- `not_mapped`: no candidate category or rule is assigned. This is the current
  status of the source-negative records.

The CSV currently pairs `basis_identified` with `medium`, `conditional` and
`internal_policy_required` with `low`, and `not_mapped` with an empty confidence.
This describes the existing records, not an automatic conversion implemented by
EDA. EDA reads confidence from the CSV.

## Annotation rules

1. Preserve the source label separately from the observation and interpretation.
   `no_violation` means the supplied annotation reported no violation; it does not
   mean that no action occurred or that every possible violation was checked.
2. Describe what is visible. A check not visible in the frame is not proof that
   the check never happened. An open door or an entering person does not establish
   authorization or compliance on its own. Use explicit uncertainty when needed.
3. Use the exact current CSV tags. For example, X-1 uses
   `accepted_cash_left_on_desk_after_count`, and X-11 uses
   `safe_key_left_in_lock`. The former dotted aliases are not a separate taxonomy.
   A tag is not stronger evidence than the description and its qualifications.
4. Treat the timestamp as an event marker. Do not label all nearby frames as
   violations automatically. Training on event intervals needs additional
   start/end annotation and a corresponding schema change.
5. Keep normative references conditional. This documentation synchronizes stored
   annotations; it does not establish that the cited provisions are correct,
   current, or applicable to the scene.
6. Edit the CSV first, regenerate its documentation snapshot, then rerun EDA.
   EDA never reconstructs annotations from these Markdown documents.

## Related files and EDA outputs

- `data/processed/video_inventory.csv`: original filenames, relative paths,
  normalized IDs, file hashes, technical parameters, and decoding checks.
- `data/processed/events_enriched.csv`: eligible annotations joined with current
  video metadata; it is generated and must not replace the editable source CSV.
- `data/processed/validation_issues.csv`: invalid labels/timestamps, missing or
  unreadable videos, duplicate identifiers, exact copies, and other warnings.
- `data/annotations/scenes.csv`: optional camera/session/scene grouping input,
  independent of event labels. Its fields are `clip_id`, `camera_id`, `session_id`,
  `scene_group`, and `reviewed`.
- `data/processed/split_proposal.csv`: proposed train/validation assignment for
  eligible videos, with related scenes and exact copies kept together. This is
  unrelated to the word "split" in `video-labels-split.md`, which refers to
  separating observed event types from candidate violation types.

The annotation CSV is explicitly allowed by `.gitignore` so it can be versioned
with the code. Other annotation files remain ignored by default. The readable
Markdown snapshot is documentation, not an automatic fallback data source for
the notebook.
