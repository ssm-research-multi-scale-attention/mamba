Split-timescale Mamba2. Самый сильный результат дала не внешняя multi-scale ветка поверх Mamba2, а внутренняя multi-timescale инициализация самих SSM-heads. В обычной Mamba2 все heads инициализируются из одного диапазона A/dt, тогда как в split-timescale варианте часть heads получает fast dynamics, а часть — slow dynamics. Forward computation, число параметров и latency при этом не меняются.

Лучший вариант — 50/50 fast/slow heads. Он сохранил то же число параметров, что и Mamba2 depth4 (1.70M), но улучшил LM loss и почти насытил transition MQAR:

Mamba2 depth4:
  LM loss       1.5920
  trans704 EM   0.565
  trans768 EM   0.297
Split-timescale 50/50:
  LM loss       1.5786
  trans704 EM   0.970
  trans768 EM   0.971

Это существенно сильнее внешнего MS-gated baseline, который давал trans704 EM≈0.90 и trans768 EM≈0.48, но требовал отдельной slow-ветки и большего числа параметров.

Вывод: multi-timescale inductive bias оказался наиболее эффективен, когда вводится внутрь Mamba2 state dynamics, а не как внешняя slow-stream надстройка. Разделение heads на fast/slow временные масштабы задаёт полезную специализацию: fast heads лучше обрабатывают локальную динамику, slow heads устойчивее хранят дальние ассоциации. Это даёт сильный выигрыш на associative recall без роста параметров и без изменения inference graph.