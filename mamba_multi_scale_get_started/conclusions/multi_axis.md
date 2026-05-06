3-scale / multi-axis вариант не улучшил лучший bi-scale MS-gated baseline.

Он дал небольшой плюс по LM loss относительно MS-gated:

MS-gated            1.5758
3-scale s2/s8       1.5727

но проиграл по главному recall-сигналу MQAR:

MQAR trans704 EM:
  MS-gated          0.900
  3-scale s2/s8     0.586
MQAR trans768 EM:
  MS-gated          0.480
  3-scale s2/s8     0.000

и оказался медленнее:

batch16 tokens/s:
  MS-gated          4.25M
  3-scale s2/s8     3.79M

Вывод: простое добавление третьей временной оси не даёт автоматического улучшения. Польза multi-timescale архитектуры зависит не от числа scales, а от правильного fusion и выбора масштаба.
