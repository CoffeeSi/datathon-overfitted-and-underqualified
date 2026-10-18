"""Reproducible action-vs-visible-sequence study; preserve the fixed split and weights.

Reuses existing trainers. The candidate recognizes one visual cash sequence,
not all legal violations. Validation is a previously inspected development set.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import yaml

from .action_evaluation import evaluate_actions
from .action_train import read_config as read_action_config, write_json
from .banknote_evaluation import evaluate_banknote_pipeline, summarize_grid_sequences
from .banknote_model import read_config as read_banknote_config, train as train_banknotes
from .dataset import load_records, sha256
from .primitive_model import read_config as read_primitive_config, run_primitive_training
from .train import ROOT, seed_everything


def checked_inputs(config):
    """Fingerprint immutable sources and both selected checkpoints before any fit."""
    result = {}
    for key in ("events_path", "split_path", "baseline_checkpoint", "binary_checkpoint"):
        path = ROOT / config[key]
        digest = sha256(path)
        expected = config.get(key.removesuffix("_path") + "_sha256")
        if expected is not None and digest != expected:
            raise ValueError(f"Pinned input changed: {key}")
        result[key] = {"path": str(path), "sha256": digest}
    return result


def verify_unchanged(inputs):
    for key, value in inputs.items():
        if sha256(Path(value["path"])) != value["sha256"]:
            raise ValueError(f"Experiment changed an immutable input: {key}")


def write_stage(template, path, *, output, config, **extra):
    stage = yaml.safe_load((ROOT / template).read_text(encoding="utf-8"))
    stage.update({key: config[key] for key in (
        "seed", "device", "cpu_threads", "events_path", "split_path", "video_dir",
        "events_sha256", "split_sha256")})
    stage.update(output_dir=str(output), **extra)
    path.write_text(yaml.safe_dump(stage, sort_keys=False), encoding="utf-8")
    return path


def compare_cash_sequences(records, source, baseline_scans, candidate_scans, tolerance_sec):
    """Identical reviewed video/grid denominator and tolerance for both pipelines."""
    validation = [r for r in records if r.split == "validation"]
    baseline = [dict(clip_id=r.clip_id, video_sha256=baseline_scans[r.clip_id]["sha256"],
                     events=baseline_scans[r.clip_id]["typed_events"]) for r in validation]
    candidate = [candidate_scans[r.clip_id] for r in validation]
    for record, scan in zip(validation, candidate):
        if scan["video_sha256"] != record.sha256:
            raise ValueError("Candidate video hash differs from fixed split")
    return {
        "baseline": summarize_grid_sequences(validation, baseline, source, tolerance_sec),
        "candidate": summarize_grid_sequences(validation, candidate, source, tolerance_sec),
        "scope": "unsecured_accepted_cash visual proxy, complete reviewed one-second grids only; not all legal types",
        "independent_evaluation": False,
    }


def write_report(path, result):
    comparison = result["comparison"]
    lines = ["# Улучшение распознавания действий: тип и время", "",
        "Текущая R3D action-голова сравнивается с обученной системой видимых признаков: "
        "двухракурсный R3D-18 (клиент/передача денег), Tiny U-Net (купюры на столе), "
        "проверка изображения между детекциями и причинная временная цепочка.", "",
        "## Один и тот же протокол", "",
        "Полностью обработаны все 10 validation-видео, split 30/10 не менялся. "
        "Video-level метрики ниже относятся только к четырём полным просмотренным "
        "секундным сеткам одного визуального сценария наличных. Другие шесть видео "
        "остаются unknown. Это не accuracy всех нарушений и не независимый test.", "",
        "Настройки взяты из ранее исследованных протоколов и зафиксированы до запуска: "
        "60 эпох R3D-головы, 60+30 эпох детектора; threshold 0.7, dwell 2 s, шаг 1 s. "
        "Ранняя остановка и подбор порога на validation не применялись. "
        "Эта validation ранее многократно просматривалась; повторение исторической "
        "архитектуры не считается новой независимой проверкой.", "", "## Сравнение", ""]
    for name, label in (("baseline", "Прежняя R3D action-модель"), ("candidate", "Новая визуальная последовательность")):
        metrics = comparison[name]["video"]
        temporal = comparison[name]["temporal"]
        lines += [f"- **{label}:** Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, "
                  f"F1={metrics['f1']:.3f}, accuracy={metrics['accuracy']:.3f}; "
                  f"confusion matrix [TN FP; FN TP] = {metrics['confusion_matrix']}. "
                  f"События: найдено {temporal['matched']}, пропущено {temporal['missed']}, "
                  f"лишних {temporal['unmatched_predictions']}; "
                  f"MAE времени на найденных = {temporal['timestamp_mae_sec']} s (допуск ±{temporal['tolerance_sec']} s)."]
    lines += ["", "## Разбор видео", ""]
    baseline = {v["clip_id"]: v for v in comparison["baseline"]["videos"]}
    for video in comparison["candidate"]["videos"]:
        previous = baseline[video["clip_id"]]
        lines.append(f"- {video['clip_id']}: прежняя {previous['outcome']} @ {previous['predicted_times_sec']}; "
                     f"новая {video['outcome']} @ {video['predicted_times_sec']}; "
                     f"визуальная proxy-отметка {video['reference_times_sec']}.")
    markers = result["candidate_metrics"]["known_action_markers"]
    lines += ["", f"Отдельно по action-отметкам из events.csv: найдено {markers['matched']}, "
              f"пропущено {markers['missed']}; MAE={markers['timestamp_mae_sec']} s. "
              "Визуальная proxy-отметка и исходное время события — разные основания; "
              "ни одно из них не переписывалось.", "", "## Ограничения", "",
        "- Поддерживается только кандидат `unsecured_accepted_cash` с визуальной цепочкой. "
        "Проверка личности и остальные типы этим вариантом не распознаются; они не считаются отрицательными.",
        "- При отсутствии полной цепочки результат — недостаточно данных, а не доказанное отсутствие нарушения.",
        "- Идентичность и принадлежность наличных, завершение учёта и применимость нормы по кадрам не установлены. "
        "Ссылка на каталог условная, requires_review=true; юридические Precision/F1 остаются недоступны.",
        "- Один положительный validation-ролик и общий кабинет не подтверждают переносимость.",
        "- Метрики отдельных примитивов не равны качеству типа события; полная временная оценка приведена отдельно.",
        "", "## Артефакты и сохранность", "",
        f"- Новые веса: `{result['checkpoints']['primitives']}` и `{result['checkpoints']['banknotes']}`.",
        f"- Отчёт baseline: `{result['baseline_report']}`; отчёт кандидата и числовой ZIP: `{result['candidate_report']}`.",
        "- Исходные веса binary/action, events.csv и split проверены SHA-256 до и после; байты сохранены.",
        "- Генерируемые manifest и конфиги — происхождение эксперимента, не новый источник разметки.",
        "- Активный backend автоматически не переключён. Коммит/push не выполнялись.",
        "", "Повторение с новым `output_dir`: `PYTHONPATH=src python -m overfitted_underqualified.ml.action_improvement "
        "--config configs/action_improvement.yaml`. Настройки каждой стадии сохранены рядом с новыми весами."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(config_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    if output.exists():
        raise FileExistsError("Use a fresh action improvement directory")
    inputs = checked_inputs(config)
    records = load_records(ROOT / config["events_path"], ROOT / config["split_path"], ROOT / config["video_dir"])
    output.mkdir(parents=True, exist_ok=False)
    report_dir = ROOT / "reports" / ("evaluation_" + output.name)
    if report_dir.exists():
        raise FileExistsError(report_dir)
    write_json(output / "plan.json", dict(config=config, inputs=inputs,
        selection="Fixed historical training protocols, last epochs and thresholds; no validation search",
        reviewed_grid_was_used_in_previous_research=True, independent_evaluation=False))
    seed_everything(config["seed"], config["cpu_threads"])

    print("BASELINE_EVALUATION", flush=True)
    baseline_config = read_action_config(ROOT / config["baseline_config"])
    baseline_config.update(device=config["device"], cpu_threads=config["cpu_threads"])
    baseline_report, baseline_metrics = evaluate_actions(baseline_config,
        ROOT / config["binary_checkpoint"], ROOT / config["baseline_checkpoint"],
        baseline_config["target_types"], tolerance_sec=config["match_tolerance_sec"])
    write_json(output / "baseline_evaluation.json", {"report_dir": str(baseline_report)})
    torch.cuda.empty_cache()

    primitive_config = write_stage(config["primitive_template"], output / "primitives.yaml",
        output=output / "primitives", config=config)
    print("TRAIN_PRIMITIVES", flush=True)
    primitive_checkpoint = run_primitive_training(read_primitive_config(primitive_config))
    verify_unchanged(inputs)
    torch.cuda.empty_cache()

    stage1_config = write_stage(config["banknote_stage1_template"], output / "banknotes_stage1.yaml",
        output=output / "banknotes_stage1", config=config)
    print("TRAIN_BANKNOTES_STAGE1", flush=True)
    stage1_checkpoint = train_banknotes(read_banknote_config(stage1_config))
    verify_unchanged(inputs)
    torch.cuda.empty_cache()

    stage2_config = write_stage(config["banknote_stage2_template"], output / "banknotes_stage2.yaml",
        output=output / "banknotes_stage2", config=config,
        initialize_checkpoint=str(stage1_checkpoint), initialize_checkpoint_sha256=sha256(stage1_checkpoint))
    print("TRAIN_BANKNOTES_STAGE2", flush=True)
    candidate_checkpoint = train_banknotes(read_banknote_config(stage2_config))
    verify_unchanged(inputs)
    torch.cuda.empty_cache()

    print("CANDIDATE_EVALUATION", flush=True)
    candidate_metrics = evaluate_banknote_pipeline(primitive_checkpoint, candidate_checkpoint,
        ROOT / config["events_path"], ROOT / config["split_path"], ROOT / config["video_dir"],
        report_dir, config["device"], track_banknotes=config["track_banknotes"])
    baseline_scans = json.loads((baseline_report / "scans.json").read_text(encoding="utf-8"))
    candidate_scans = {r.clip_id: json.loads((report_dir / f"{r.clip_id}_scan.json").read_text(encoding="utf-8"))
                       for r in records if r.split == "validation"}
    with (ROOT / config["events_path"]).open(encoding="utf-8-sig", newline="") as stream:
        source = {r["clip_id"]: r for r in csv.DictReader(stream)}
    comparison = compare_cash_sequences(records, source, baseline_scans, candidate_scans,
                                       config["match_tolerance_sec"])
    verify_unchanged(inputs)
    result = dict(schema="action_improvement_study_v1", inputs=inputs, config=config,
        checkpoints={"primitives": str(primitive_checkpoint), "banknotes": str(candidate_checkpoint)},
        checkpoint_sha256={"primitives": sha256(primitive_checkpoint), "banknotes": sha256(candidate_checkpoint)},
        baseline_report=str(baseline_report), candidate_report=str(report_dir),
        baseline_window_metrics=baseline_metrics["window_metrics"],
        baseline_temporal=baseline_metrics["temporal"], candidate_metrics=candidate_metrics,
        comparison=comparison, baseline_weights_unchanged=True, split_and_events_unchanged=True,
        active_service_replaced=False, independent_evaluation=False,
        limitation="Candidate supports a visual cash sequence only. Identity-check absence and other legal types are unsupported, not proved negative.")
    write_json(output / "comparison.json", result)
    summary_path = ROOT / "reports" / (output.name + ".json")
    if summary_path.exists():
        raise FileExistsError(summary_path)
    write_json(summary_path, result)
    write_report(summary_path.with_suffix(".md"), result)
    print("STUDY_SAVED " + str(summary_path), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/action_improvement.yaml")
    args = parser.parse_args()
    run_study(args.config)
