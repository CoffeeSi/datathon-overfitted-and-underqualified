# План очистки старых артефактов

Статус: только план; удаление не выполнено. Автоматическая проверка отклонила массовое необратимое удаление без согласования точного списка.

Будут сохранены данные, код, все краткие отчёты исследований и metadata/history прежних запусков. Исторические пути удалённых весов останутся в отчётах как provenance.

## Сохранить веса

- `runs/binary_improvements_20261017T230201_728607Z/roi/best.pt`
- `runs/r3d18_augmented_20261018T023826_596184Z/best.pt`
- `runs/r3d18_augmented_20261018T023826_596184Z/last.pt`
- `runs/unsecured_cash_20261018T011640_619812Z/last.pt`
- `runs/r3d18_actions_balanced_20261018T004011Z/last.pt`
- `runs/primitives_multiview_20261018T013815_768128Z/last.pt`
- `runs/banknotes_balanced_20261018T021500Z/last.pt`
- `runs/banknotes_finetune_20261018T021538_229103Z/last.pt`

## Удалить: 
superseded_checkpoint

- `runs/banknotes_augmented_20261018T020455Z/last.pt` — 0.51 MiB
- `runs/banknotes_hard_negatives_20261018T022500Z/last.pt` — 0.51 MiB
- `runs/binary_improvements_20261017T225920_013664Z/negatives/best.pt` — 126.62 MiB
- `runs/binary_improvements_20261017T225920_013664Z/negatives/last.pt` — 126.62 MiB
- `runs/binary_improvements_20261017T230201_728607Z/control_stable/best.pt` — 126.61 MiB
- `runs/binary_improvements_20261017T230201_728607Z/control_stable/last.pt` — 126.61 MiB
- `runs/binary_improvements_20261017T230201_728607Z/geometry/best.pt` — 126.62 MiB
- `runs/binary_improvements_20261017T230201_728607Z/geometry/last.pt` — 126.62 MiB
- `runs/binary_improvements_20261017T230201_728607Z/negatives/best.pt` — 126.62 MiB
- `runs/binary_improvements_20261017T230201_728607Z/negatives/last.pt` — 126.62 MiB
- `runs/binary_improvements_20261017T230201_728607Z/repeat_roi/best.pt` — 126.62 MiB
- `runs/binary_improvements_20261017T230201_728607Z/repeat_roi/last.pt` — 126.62 MiB
- `runs/binary_improvements_20261017T230201_728607Z/roi/last.pt` — 126.62 MiB
- `runs/binary_improvements_20261017T230201_728607Z/roi_context8/best.pt` — 126.62 MiB
- `runs/binary_improvements_20261017T230201_728607Z/roi_context8/last.pt` — 126.62 MiB
- `runs/identity_control_20261018T005722_203342Z/last.pt` — 126.66 MiB
- `runs/identity_roi_20261018T005722_203342Z/last.pt` — 126.66 MiB
- `runs/primitives_20261018T013342_547979Z/last.pt` — 126.69 MiB
- `runs/r3d18_actions_owner_20261018T004011_430311Z/last.pt` — 126.7 MiB
- `runs/r3d18_full_video_20261017T201156_904223Z/best.pt` — 126.61 MiB
- `runs/r3d18_full_video_20261017T201156_904223Z/last.pt` — 126.61 MiB
- `runs/r3d18_notebook_20261017T223905_499373Z/best.pt` — 126.61 MiB
- `runs/r3d18_notebook_20261017T223905_499373Z/last.pt` — 126.61 MiB

## Удалить: 
old_detailed_report

- `reports/evaluation_actions_20261018T004341Z_efe9e4b9` — 0.58 MiB
- `reports/evaluation_actions_20261018T010131Z_dc1b498e` — 0.51 MiB
- `reports/evaluation_actions_20261018T010555Z_f1037c92` — 0.51 MiB
- `reports/evaluation_full_r3d18_full_video_20261017T201156_904223Z_20261017T201511Z_f0b8a3fb` — 0.04 MiB
- `reports/evaluation_full_r3d18_full_video_20261017T201156_904223Z_20261017T201706Z_546d55bf` — 0.04 MiB
- `reports/evaluation_full_r3d18_notebook_20261017T223905_499373Z_20261017T224742Z_65575f00` — 0.04 MiB
- `reports/evaluation_full_roi_20261017T231250Z_96d873d0` — 0.04 MiB
- `reports/evaluation_improvements_binary_improvements_20261017T225920_013664Z_baseline` — 0.04 MiB
- `reports/evaluation_improvements_binary_improvements_20261017T230201_728607Z_control_stable` — 0.04 MiB
- `reports/evaluation_improvements_binary_improvements_20261017T230201_728607Z_geometry` — 0.04 MiB
- `reports/evaluation_improvements_binary_improvements_20261017T230201_728607Z_negatives` — 0.04 MiB
- `reports/evaluation_improvements_binary_improvements_20261017T230201_728607Z_roi_context8` — 0.04 MiB
- `reports/evaluation_primitives_20261018T013342Z` — 0.25 MiB
- `reports/evaluation_r3d18_full_video_20261017T201156_904223Z_20261017T201412Z_630a46b3` — 0.05 MiB
- `reports/evaluation_r3d18_full_video_20261017T201156_904223Z_20261017T224223Z_1648b3d1` — 0.05 MiB
- `reports/evaluation_r3d18_notebook_20261017T223905_499373Z_20261017T224642Z_11b30d9c` — 0.05 MiB
- `reports/evaluation_roi_20261017T231148Z_0b8b895c` — 0.08 MiB

## Удалить: 
old_report_archive

- `reports/evaluation_actions_20261018T004341Z_efe9e4b9.zip` — 0.07 MiB
- `reports/evaluation_actions_20261018T010131Z_dc1b498e.zip` — 0.06 MiB
- `reports/evaluation_actions_20261018T010555Z_f1037c92.zip` — 0.06 MiB
- `reports/evaluation_full_r3d18_full_video_20261017T201156_904223Z_20261017T201511Z_f0b8a3fb.zip` — 0.01 MiB
- `reports/evaluation_full_r3d18_full_video_20261017T201156_904223Z_20261017T201706Z_546d55bf.zip` — 0.01 MiB
- `reports/evaluation_full_r3d18_notebook_20261017T223905_499373Z_20261017T224742Z_65575f00.zip` — 0.01 MiB
- `reports/evaluation_full_roi_20261017T231250Z_96d873d0.zip` — 0.01 MiB
- `reports/evaluation_improvements_binary_improvements_20261017T225920_013664Z_baseline.zip` — 0.01 MiB
- `reports/evaluation_improvements_binary_improvements_20261017T230201_728607Z_control_stable.zip` — 0.01 MiB
- `reports/evaluation_improvements_binary_improvements_20261017T230201_728607Z_geometry.zip` — 0.01 MiB
- `reports/evaluation_improvements_binary_improvements_20261017T230201_728607Z_negatives.zip` — 0.01 MiB
- `reports/evaluation_improvements_binary_improvements_20261017T230201_728607Z_roi_context8.zip` — 0.01 MiB
- `reports/evaluation_primitives_20261018T013342Z.zip` — 0.03 MiB
- `reports/evaluation_r3d18_full_video_20261017T201156_904223Z_20261017T201412Z_630a46b3.zip` — 0.02 MiB
- `reports/evaluation_r3d18_full_video_20261017T201156_904223Z_20261017T224223Z_1648b3d1.zip` — 0.02 MiB
- `reports/evaluation_r3d18_notebook_20261017T192040_577572Z_20261017T194434Z_cfacebbf.zip` — 0.02 MiB
- `reports/evaluation_r3d18_notebook_20261017T223905_499373Z_20261017T224642Z_11b30d9c.zip` — 0.02 MiB
- `reports/evaluation_r3d18_smoke_20261017T191407Z_fefdcafb.zip` — 0.02 MiB
- `reports/evaluation_roi_20261017T231148Z_0b8b895c.zip` — 0.02 MiB

Освободится: 
2.6
 GiB.
Данные raw, events.csv, split, pretrained backbone и действующие настройки не изменяются.
