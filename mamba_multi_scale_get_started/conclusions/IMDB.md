📊 Итоговая таблица (IMDB)

🧠 Основные модели

| Модель | Accuracy | F1 macro | Params | Время train | Комментарий |
| - | - | - | - | - | - |
LSTM | 0.8397 | 0.8394 | 8.9M | ~? | baseline
GRU | 0.8228 | 0.8220 | 8.6M | ~14s | хуже LSTM
Mamba2 (baseline) | 0.8637 | 0.8634 | 11.1M | ~44s | лучший single-scale

⸻

🔧 Ablation: архитектура Mamba

| Конфиг | Accuracy | Вывод |
| - | - | - |
4 слоя | 0.8609 | почти как baseline → можно упрощать |
8 слоёв | 0.8637 | оптимум
12 слоёв | 0.8360 | переобучение / деградация
d_model = 128 | 0.7810 | слишком маленькая модель

👉 Вывод:

* модель чувствительна к размеру
* больше ≠ лучше
* есть sweet spot

⸻

🔀 Multi-scale (без attention)

Конфиг | Accuracy | Вывод
| - | - | - |
stride=4, concat | 0.8244 | хуже baseline
stride=2 | 0.8253 | чуть лучше, но всё равно хуже
stride=8 | 0.8118 | сильно хуже
gated | 0.8463 | лучше, но всё ещё хуже baseline

👉 Вывод:

* multi-scale сам по себе не работает
* downsampling ломает сигнал

⸻

🔥 Multi-scale + Attention

Конфиг | Accuracy | Вывод
| - | - | - |
stride=4, residual | 0.8429 | заметно лучше, но не топ
stride=4, concat | 0.8582 | почти baseline
stride=2, residual | 0.8634 | ≈ baseline
stride=2, concat | 0.8603 | стабильно хорошо

👉 Вывод:

* attention критичен
* stride=2 — лучший вариант
* concat > residual

⸻

🧪 Attention + Gated

Конфиг | Accuracy | Вывод
- | - | -
attention + gated | 0.8616 | ≈ baseline
gated (без attention) | 0.8463 | хуже

👉 Вывод:

* gated ≈ нейтрально
* attention уже делает всю работу

⸻

📈 Финальный рейтинг моделей

Место | Модель | Accuracy
- | - | -
🥇 | Mamba2 baseline | 0.8637
🥈 | Multi-scale + attention (stride=2) | ~0.8634
🥉 | Multi-scale + attention + gated | ~0.8616
4 | Multi-scale (best) | ~0.846
5 | LSTM | 0.8397
6 | GRU | 0.8228

⸻

🧠 Главные выводы (это самое важное)

1. Mamba > RNN

Mamba2 стабильно лучше LSTM/GRU (~+2–4%)

⸻

2. Multi-scale НЕ даёт прироста

Сам по себе multi-scale ухудшает качество

Причина:

* теряется информация при downsampling
* задача не требует multi-timescale

⸻

3. Attention — ключевой компонент

Attention восстанавливает качество multi-scale

Фактически:

выигрыш даёт attention, а не multi-scale

⸻

4. Optimal configuration

- 8 слоёв
- d_model = 256
- batch size ↑ помогает
- epochs > 3 полезно

⸻

5. Задача не раскрывает multi-scale

IMDB:

* короткие тексты
* мало сложной временной структуры

👉 multi-scale просто не нужен

⸻

🎯 Главный high-level вывод

Можно формулировать так:

На задаче классификации текстов (IMDB) Mamba2 превосходит классические RNN.
Попытка добавить multi-scale архитектуру не приводит к улучшению качества.
Основной вклад в улучшение даёт механизм attention, а не multi-scale представление.

⸻

🚀 Что это значит стратегически

Ты уже доказал:

✅ Mamba работает
❌ multi-scale не даёт буста здесь
✅ attention полезен

👉 логичный следующий шаг:

перейти к generative / sequence modeling задачам,
где multi-timescale действительно важен
