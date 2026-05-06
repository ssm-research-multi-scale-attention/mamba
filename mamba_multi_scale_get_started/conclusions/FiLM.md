FiLM оказался неудачным архитектурным вариантом.

С film_init_zero=true модель коллапсировала даже на easy MQAR. После отключения zero-init (film_init_zero=false) easy-режим ожил и стал почти идеальным:

easy MQAR:
  acc = 0.9997
  EM  = 0.9955

Но transition-режим остался около random:

trans704:
  acc ≈ 0.003–0.005
  EM  = 0.000

При этом train accuracy росла, а val/test оставались near-random, то есть модель переобучалась и не обобщала associative recall.

Вывод: cross-scale FiLM может работать как sanity/easy mechanism, но не даёт устойчивого recall-преимущества в transition MQAR. В текущем виде FiLM хуже MS-gated и не стоит дальнейшего развития как основной архитектурный кандидат.
