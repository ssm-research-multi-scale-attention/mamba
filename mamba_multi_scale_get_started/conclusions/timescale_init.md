Timescale initialization. Дополнительный MQAR failure sweep показал, что основной recall-выигрыш в Mamba-like моделях связан не столько с внешними multi-scale ветками, сколько с инициализацией временных масштабов внутри SSM dynamics.

Обычная Mamba2 с default initialization нестабильна уже в transition-зоне MQAR. Split fast/slow heads улучшает ситуацию и остаётся стабильным до vocab=768, но коллапсирует на vocab=1024. Однако простой slow-biased single init для всех heads оказался сильнее: Mamba2 slow-init остаётся стабильной до vocab=1024 и коллапсирует только на vocab=1280.

Failure sweep:
  Mamba2 split fast/slow     stable ≤768,  collapse at 1024
  Mamba2 slow-init           stable ≤1024, collapse at 1280
  MS-gated right/slow-init   stable ≤896,  collapse at 1280
  Transformer                stable through 2048

Wide timescale initialization (A=[0.1,32], wide dt) оказалась нестабильной уже на лёгких/средних режимах, что показывает: полезно не просто расширить диапазон временных масштабов, а именно сдвинуть SSM dynamics к более медленной памяти.

Вывод: лучший recurrent recall результат даёт не внешняя slow-stream архитектура и не fast/slow split heads, а простая slow-biased initialization Mamba2. Это говорит о том, что MQAR recall в Mamba2 сильно зависит от начального timescale prior. Multi-scale ветки могут помогать относительно default Mamba2, но после правильной SSM initialization они не дают явного преимущества над более простой Mamba2 slow-init. Transformer при этом остаётся upper baseline для explicit retrieval, стабильно решая MQAR во всём проверенном диапазоне до vocab=2048.