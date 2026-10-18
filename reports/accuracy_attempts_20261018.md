# Попытки улучшить accuracy на текущих видео

Улучшение не достигнуто. Прежняя модель с validation accuracy 60% и F1=0.50 сохранена. Три новых кандидата измерены, но не включены в backend и не выбраны в .env.

Две фазы: заранее заданные регуляризованные головы, затем отдельно объявленный опыт временных сдвигов после неудачи первой фазы. Внутри фаз параметры/порог по validation не подбирались. Это development-сравнение, а не независимое подтверждение качества.

## baseline

- Train (30 исходников): accuracy 80.0%, F1 0.824.
- Validation (10 исходников): accuracy 60.0%, Precision 0.667, Recall 0.400, F1 0.500.
- Матрица [[TN,FP],[FN,TP]]: [[4, 1], [3, 2]].
- 40 связанных преобразований: accuracy 65.0%, F1 0.533.
- Checkpoint: `D:\CodeSpace\datathon\datathon-overfitted-and-underqualified\runs\r3d18_matched_20261018T025716_174675Z\mild_augmented\last.pt`.

## l2_logistic

- Train (30 исходников): accuracy 100.0%, F1 1.000.
- Validation (10 исходников): accuracy 50.0%, Precision 0.500, Recall 0.200, F1 0.286.
- Матрица [[TN,FP],[FN,TP]]: [[4, 1], [4, 1]].
- 40 связанных преобразований: accuracy 45.0%, F1 0.312.
- Checkpoint: `D:\CodeSpace\datathon\datathon-overfitted-and-underqualified\runs\r3d18_regularized_20261018T035417_731553Z\l2_logistic\last.pt`.

## shrinkage_lda

- Train (30 исходников): accuracy 100.0%, F1 1.000.
- Validation (10 исходников): accuracy 40.0%, Precision 0.333, Recall 0.200, F1 0.250.
- Матрица [[TN,FP],[FN,TP]]: [[3, 2], [4, 1]].
- 40 связанных преобразований: accuracy 45.0%, F1 0.312.
- Checkpoint: `D:\CodeSpace\datathon\datathon-overfitted-and-underqualified\runs\r3d18_regularized_20261018T035417_731553Z\shrinkage_lda\last.pt`.

## temporal_jitter

- Train (30 исходников): accuracy 86.7%, F1 0.875.
- Validation (10 исходников): accuracy 50.0%, Precision 0.500, Recall 0.400, F1 0.444.
- Матрица [[TN,FP],[FN,TP]]: [[3, 2], [3, 2]].
- 40 связанных преобразований: accuracy 57.5%, F1 0.452.
- Checkpoint: `D:\CodeSpace\datathon\datathon-overfitted-and-underqualified\runs\r3d18_temporal_jitter_20261018T040124_278836Z\last.pt`.

## Вывод

На train новые головы улучшили разделение, но на validation потеряли качество. Временная аугментация также не дала прироста. Эти результаты поддерживают вывод об ограничении переноса на текущих данных; они не доказывают конкретную причину ошибок и не оправдывают подгонку меток или порога по десяти известным видео.

Split 30/10, исходные метки и таймстампы не менялись. Для каждого временного сдвига проверено сохранение известной отметки внутри окна. X-4/X-47 сохраняют два события одного канонического видео. Нет новых независимых положительных примеров.

148 тестов прошли; новые веса восстановлены стандартным load_checkpoint и оценены. Ранние признаки/BatchNorm, прежние веса и исходные хеши сохранены. Outputs ноутбуков пустые.

[Фаза регуляризации](regularized_study_20261018T035417_731553Z.md), [фаза временных сдвигов](temporal_jitter_study_20261018T040124_278836Z.md), [график](evaluation_accuracy_attempts_20261018/comparison.png).
