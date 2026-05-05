A. Context scaling
  На Tiny Shakespeare все модели улучшаются при росте block_size 256 → 512 → 1024.
  MS-варианты лучше маленькой Mamba2 depth=4 на всех длинах.
  Лучший raw результат: MS-attention.

B. Stride sweep
  Для MS-gated лучший stride = 2:
    stride=2:  loss 1.5730, PPL 4.8213
    stride=4:  loss 1.5751, PPL 4.8312
    stride=8:  loss 1.5755, PPL 4.8329
    stride=16: loss 1.5750, PPL 4.8309

  Но разница очень маленькая.
  Вывод: gated MS почти не чувствителен к stride на Tiny Shakespeare.

C. Fusion sweep
  На block_size=1024, stride=4:

    MS-attention: loss 1.5656, PPL 4.7854
    MS-concat:    loss 1.5739, PPL 4.8256
    MS-gated:     loss 1.5751, PPL 4.8312
    MS-sum:       loss 1.5772, PPL 4.8412

  Лучший fusion: attention.
  Среди простых fusion concat/gated/sum почти одинаковы.

D. Timing
  Mamba2 быстрее всех.
  Multi-scale стоит примерно +50% latency.

  batch=1, block_size=1024:
    Mamba2:        ~2.35 ms
    MS-gated s=4:  ~3.50 ms
    MS-gated s=2:  ~3.60 ms
    MS-attention:  ~3.65 ms

  Вывод: MS даёт небольшой выигрыш по PPL над маленькой Mamba2,
  но стоит дороже по latency и параметрам.

E. Equal-param Mamba2 baseline
  Это главный результат.

  Mamba2 depth=6, 2.54M params:
    loss 1.5370, PPL 4.6506

  Mamba2 depth=7, 2.95M params:
    loss 1.5288, PPL 4.6129

  MS-gated stride=2, 2.67M params:
    loss 1.5730, PPL 4.8213

  MS-attention stride=4, 2.80M params:
    loss 1.5656, PPL 4.7854

  Вывод: текущие MS выигрывают у маленькой Mamba2 depth=4,
  но проигрывают equal-param Mamba2.

На обычном LM-quality для Tiny Shakespeare текущая naive multi-scale архитектура не даёт архитектурного выигрыша.
Её преимущество над маленькой Mamba2 объясняется увеличением числа параметров.
При честном сравнении лучше просто углубить Mamba2.

MS-attention — лучший из multi-scale вариантов, но всё равно хуже equal-param Mamba2.
