# Report

## Track

Выбранный трек:

```text
В
```

## Что реализовано

- [x] dataset.py
- [x] processor.py
- [x] model.py
- [x] train.py
- [x] benchmark.py

## Конфигурация

```text
config path:
seed:
device:
dtype:
max_steps:
batch size:
```

## Результаты

```text
public tests: 
❯ python -m pytest tests_public/ -q
..............                                                                                                                                                                                             [100%]
14 passed in 3.73s

train loss:

❯ python -m hw.train --config configs/track_b_small_gpu_medium.yaml
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 198/198 [00:00<00:00, 4303.88it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [00:00<00:00, 3165.05it/s]
Added image token, new vocab size: 151647
Image token ID: 151646
Step 0/300, loss: 1.1490
Step 5/300, loss: 1.8111
Step 10/300, loss: 2.2572
Step 15/300, loss: 1.3847
Step 20/300, loss: 1.1243
Step 25/300, loss: 7.7550
Step 30/300, loss: 1.7300
Step 35/300, loss: 1.3220
Step 40/300, loss: 0.7232
Step 45/300, loss: 1.1338
Step 50/300, loss: 1.7827
Step 55/300, loss: 2.0550
Step 60/300, loss: 1.6180
Step 65/300, loss: 3.5067
Step 70/300, loss: 1.3034
Step 75/300, loss: 1.8634
Step 80/300, loss: 1.0961
Step 85/300, loss: 1.4811
Step 90/300, loss: 1.9075
Step 95/300, loss: 1.6710
Step 100/300, loss: 3.2954
Step 105/300, loss: 0.9751
Step 110/300, loss: 3.1544
Step 115/300, loss: 1.8999
Step 120/300, loss: 1.2501
Step 125/300, loss: 1.6461
Step 130/300, loss: 0.4979
Step 135/300, loss: 0.7761
Step 140/300, loss: 2.7411
Step 145/300, loss: 2.6819
Step 150/300, loss: 3.4140
Step 155/300, loss: 1.4075
Step 160/300, loss: 0.8283
Step 165/300, loss: 1.1445
Step 170/300, loss: 2.0713
Step 175/300, loss: 1.4039
Saved adapter checkpoint to checkpoints/adapter_medium.pt

benchmark accuracy:

❯ python -m hw.benchmark --config configs/inference_math.yaml
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 198/198 [00:00<00:00, 5100.77it/s]
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [00:00<00:00, 5651.34it/s]
Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:10<00:00,  2.71s/it]
{
  "overall": 0.0,
  "subject/geometry": 0.0,
  "subject/plots": 0.0
}
```

## Использованные ресурсы

```text
CPU/GPU:
VRAM:
время обучения:
```

## Анализ ошибок

Приведите 3 ошибки модели:

1. ...
2. ...
3. ...

## Комментарии

Что оказалось самым сложным, что бы вы улучшили?


## Критерии оценивания

См. файл [`GRADING.md`](GRADING.md).
