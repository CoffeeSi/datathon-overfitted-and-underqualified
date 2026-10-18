"""Summarize immutable saved primitive experiments; never run training or tune thresholds."""
import argparse
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from overfitted_underqualified.ml.action_train import write_json
from overfitted_underqualified.ml.dataset import sha256


def summarize(report):
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    checkpoint = Path(summary["checkpoint"])
    if sha256(checkpoint) != summary["checkpoint_sha256"]:
        raise ValueError("Saved report/checkpoint mismatch")
    run = checkpoint.parent
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    checks = json.loads((run / "checks.json").read_text(encoding="utf-8"))
    if any(config[k] != summary[k] for k in ("events_sha256", "split_sha256")):
        raise ValueError("Saved experiment source mismatch")
    return dict(report=report.relative_to(ROOT).as_posix(), checkpoint=checkpoint.relative_to(ROOT).as_posix(),
                checkpoint_sha256=summary["checkpoint_sha256"], config_sha256=sha256(run / "config.json"),
                config=config, train_metrics=history[-1]["train"],
                validation_metrics=summary["primitive_metrics"], temporal=summary["temporal"],
                full_scan_processing_sec=summary["full_scan_processing_sec"], total_candidates=summary["total_candidates"],
                checks=checks, errors=json.loads((report / "primitive_errors.json").read_text(encoding="utf-8")))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first", type=Path, required=True)
    p.add_argument("--second", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New JSON summary, with companion .md")
    args = p.parse_args()
    if args.output.exists() or args.output.with_suffix(".md").exists():
        raise FileExistsError("Summary already exists; historical metrics must not be rewritten")
    first, second = [summarize(path.resolve()) for path in (args.first, args.second)]
    if any(first["config"][k] != second["config"][k] for k in
           ("events_sha256", "split_sha256", "seed", "epochs", "window_sec", "threshold", "sequence")):
        raise ValueError("Experiments differ beyond view geometry")
    study = dict(schema="visible_primitive_study_v1", experiments=dict(service_view=first, service_and_desk=second),
                 events_sha256=first["config"]["events_sha256"], split_sha256=first["config"]["split_sha256"],
                 fixed_split=dict(train=30, validation=10), reviewed_points=dict(train=54, validation=39),
                 reviewed_videos=dict(train=8, validation=6), independent_evaluation=False, deployment_enabled=False,
                 p0_quality_closed=False,
                 conclusion="Client presence transfers to reviewed validation points; cash-on-counter recognition fails, so the temporal cash hypothesis remains unvalidated.")
    write_json(args.output, study)
    lines = ["# Видимые признаки и временная цепочка: 18 октября 2026", "",
             "Подход реализован и проверен двумя реальными запусками. Исследовательский scanner проходит видео с начала,",
             "восстанавливает причинную историю обслуживания и сохраняет вероятности, отказы, события-кандидаты и их основания.",
             "**Качество цепочки пока не подтверждено: оба варианта пропустили X-38. P0 по качеству не закрыта.**", "",
             "## Данные и протокол", "",
             "93 просмотренные точки в 14 канонических видео: 54 train (8 видео), 39 validation (6 видео).",
             "Полный scan выполнен на всех 10 validation-видео. Исходные поля events.csv, action_review и split 30/10 сохранены.",
             "Добавлен только primitive_review; отсутствие ключа означает unknown. Передача денег имеет 5 train-положительных",
             "примеров из 5 видео и 3 validation-положительных из 3 видео; это небольшой объём, а не полноценный новый датасет.",
             "Просмотр кадров выполнен в ходе подготовки этого исследования; независимая человеческая экспертиза не заявляется.", "",
             "Два фиксированных запуска: frozen R3D-18, 16 кадров за предыдущие 2 секунды, seed 42, 60 эпох, lr=0.001,",
             "masked BCE и train-only class weights. Общий вид [0,.12,1,.66]; во втором варианте добавлен вид стола",
             "[0,.32,.82,.56]. Порог 0.7, подтверждённое отсутствие <=0.3, dwell 2s, scan step 1s, память receipt 120s.",
             "Второй эксперимент предложен после диагностики первого; оба exploratory. Пороги не подбирались, веса",
             "сохранены с последней фиксированной эпохи. Это многократно просмотренная validation, не независимый test.", ""]
    for title, item in (("Один общий вид", first), ("Общий вид + стол", second)):
        lines += [f"## {title}", "", f"Checkpoint: `{item['checkpoint']}`.", f"SHA-256: `{item['checkpoint_sha256']}`.", ""]
        for metric in item["validation_metrics"]["per_class"]:
            lines.append(f"- {metric['event_type']}: Precision {metric['precision']:.3f}, Recall {metric['recall']:.3f}, "
                         f"F1 {metric['f1']:.3f}; [[TN,FP],[FN,TP]]={metric['confusion_matrix']}; "
                         f"известных положительных {metric['support']}, отрицательных {metric['known_negatives']}, unknown {metric['unknown']}.")
        lines += ["", f"Полный проход: {item['full_scan_processing_sec']:.2f} s CUDA; кандидатов {item['total_candidates']}; "
                  f"совпадений {item['temporal']['matched']}, пропусков {item['temporal']['missed']}.",
                  f"Параметры backbone и BatchNorm неизменны: `{item['checks']['frozen_backbone_and_bn_unchanged']}`.",
                  f"[Отчёт и ZIP](../{item['report']}.zip) (локальный архив игнорируется Git).", ""]
    lines += ["## Разбор ошибок", "",
              "У второго варианта клиент у стойки распознан на всех 39 проверенных точках, передача денег — на 2 из 3",
              "положительных окон; X-2 @12s пропущен. Это отдельные признаки, не качество распознавания нарушений.",
              "Банкноты: пропущены все 14 validation-положительных точек; ложный positive X-2 @4s на пустом столе.",
              "Проверенные изображения показывают деньги на X-2 @10s, X-10/X-14 и X-38 после ухода клиента.",
              "При этом train F1 признака банкнот около 0.963. Разрыв train/validation и высокая оценка пустого стола",
              "согласуются с переобучением на контекст; это диагностическая гипотеза, не доказательство внутреннего механизма модели.",
              "Увеличение области стола само по себе проблему не исправило. Снижением порога нельзя одновременно убрать",
              "ложный positive X-2 и восстановить все настоящие положительные точки: неверный порядок оценок уже сохранён.", "",
              "## Что теперь проверяемо", "",
              "Цепочка проверяется отдельно от визуальных предсказаний: нет receipt → нет типизированной тревоги; неизвестное",
              "состояние не считается уходом; разрыв/истечение истории сбрасывает контекст; новый эпизод допускает следующее событие.",
              "На реальных видео ни один вариант не породил полный cash-кандидат. Это не доказательство отсутствия нарушений.",
              "Полная precision/F1 по типу и timestamp MAE остаются null: временная разметка неполна и совпадений нет.",
              "Ссылки на правило берутся из существующего каталога только для исследовательского кандидата; тождество купюр,",
              "завершение операции и применимость правила этой системой не установлены. FastAPI не переключён на неё.", "",
              "Следующий обоснованный эксперимент — отдельное распознавание банкнот с объектной разметкой и разнообразными",
              "положительными/отрицательными train-примерами. Повторять обучение той же головы без изменения этой задачи нецелесообразно.", "",
              f"events SHA-256: `{study['events_sha256']}`; split SHA-256: `{study['split_sha256']}`.", "",
              "Численные результаты, checkpoint/config/data hashes, FP/FN и временные траектории находятся в companion JSON",
              "и двух ZIP. Кадры/веса в ZIP не включены; прежние action/binary результаты сохранены без переписывания."]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
