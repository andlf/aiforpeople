Что изменилось по сравнению с прошлой версией:

Каждый термин объясняется при первом использовании — токен, эмбеддинг, attention, causal, pretraining, SFT, LoRA, KV-cache, perplexity (см. «Мини-словарь» в начале).

Аналогии — библиотека с книгами (attention), вращение стрелок на циферблате (RoPE), «ворота» (SwiGLU), шардирование (FSDP).

Код с подробными комментариями — каждая строка, которая может быть неочевидна, объяснена.

Реальные файлы с именами — model/rmsnorm.py, model/rope.py, data/dataset.py, train/train_pretrain.py, finetune/lora.py, inference/kv_cache.py и т.д.

Упражнения с полными решениями — exercises/check_rmsnorm.py, ex_2_1_bpe_from_scratch.py, ex_3_1_attention_naive.py, ex_4_1_rope_visualize.py, ex_5_1_weight_tying.py, ex_6_1_bf16.py, ex_8_1_kv_cache.py.

Каждая глава заканчивается проверкой — скрипт, доказывающий, что код работает (assert).

Как запустить:

bash
# создайте проект
mkdir llm-from-scratch && cd llm-from-scratch

# сохраните этот .md в README.md или разбейте на файлы по указанным путям
# установите зависимости
pip install torch tiktoken matplotlib

# проверьте, что всё работает
python -m exercises.check_rmsnorm
python -m exercises.ex_3_1_attention_naive
python -m exercises.ex_5_1_weight_tying
