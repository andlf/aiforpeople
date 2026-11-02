---
title: "Современный LLM с нуля: от attention до Chat-модели"
author: "Open Book"
lang: ru-RU
---

# Современный LLM с нуля: от attention до Chat-модели

**Практическое руководство по Python и PyTorch. Архитектуры 2024–2026, объяснения «на пальцах», рабочий код с комментариями.**

## Как читать эту книгу

Эта книга — не справочник и не набор «кусков кода». Каждая глава строится по одному шаблону:

1. **Зачем** — какую проблему решаем.
2. **Что** — концепция простыми словами, с аналогией.
3. **Как** — рабочий код с подробными комментариями.
4. **Проверка** — короткий запускаемый скрипт, доказывающий, что код работает.
5. **Упражнение** — самостоятельная задача с полным решением в виде готового файла.

Всё, что описано, можно запустить на одном GPU или в уменьшенном масштабе на CPU. Итоговая структура проекта:

```text
llm-from-scratch/
├── config.py                 # параметры модели
├── data/
│   ├── tokenizer_demo.py     # как работает BPE
│   └── dataset.py            # датасет для pretraining и SFT
├── model/
│   ├── rmsnorm.py            # нормализация
│   ├── rope.py               # позиционное кодирование
│   ├── attention.py          # MHA, GQA, SDPA
│   ├── swiglu.py             # FFN
│   ├── block.py              # transformer-блок
│   └── gpt.py                # полная модель
├── train/
│   ├── schedule.py           # warmup + cosine
│   └── train_pretrain.py     # обучение
├── inference/
│   ├── generate.py           # генерация с top-k/top-p
│   └── kv_cache.py           # быстрая генерация
├── finetune/
│   ├── sft.py                # supervised fine-tuning
│   └── lora.py               # LoRA
└── exercises/
    ├── ex_2_1_bpe_from_scratch.py
    ├── ex_3_1_attention_naive.py
    ├── ex_4_1_rope_visualize.py
    ├── ex_5_1_weight_tying.py
    ├── ex_8_1_kv_cache.py
    └── ...
```

---

## Мини-словарь терминов

Прежде чем идти дальше — давайте договоримся о словах, которые будут встречаться сотни раз. Если вы знаете их все — переходите к главе 1.

- **Токен.** Единица текста, с которой работает модель. Обычно — часть слова: например, `"программирование"` → `["программ", "ирование"]`. Токеном может быть и один символ, и целое слово. Модель видит только числа (ID токенов), а не буквы.
- **Эмбеддинг.** Вектор чисел фиксированной длины, которым «заменяется» токен. Например, `[0.12, -0.94, 0.31, ...]`. Эмбеддинги обучаемые: в начале они случайные, но в процессе тренировки смещаются так, чтобы близкие по смыслу токены оказывались рядом в векторном пространстве.
- **Attention (внимание).** Механизм, который позволяет каждому токену «посмотреть» на все остальные и собрать из них контекст. Аналогия: вы читаете предложение и, встретив местоимение «он», оглядываетесь назад, чтобы понять, кто этот «он».
- **Transformer.** Архитектура нейросети, целиком построенная на attention. Практически все современные LLM (GPT, Llama, Qwen, Mistral) — это transformer.
- **Causal (причинная) модель.** Модель, которая при предсказании следующего токена может смотреть только назад, но не вперёд. Это обязательно для генерации: иначе модель «подглядывала» бы ответ.
- **Pretraining.** Первая стадия обучения: модель тренируется предсказывать следующий токен на огромном корпусе текста. Результат — «базовая» модель, которая знает язык и мир, но плохо следует инструкциям.
- **SFT (Supervised Fine-Tuning).** Вторая стадия: модель учится на парах «инструкция → ответ». Результат — модель, которая следует инструкциям.
- **LoRA.** Способ дообучения, при котором замораживаются все веса исходной модели, а обучаются маленькие дополнительные матрицы. Требует в десятки раз меньше памяти.
- **KV-cache.** Кэш ключей и значений при генерации, который позволяет не пересчитывать attention для уже сгенерированных токенов.
- **Perplexity.** «Запутанность» — метрика качества языковой модели. Чем меньше, тем лучше. Интерпретируется как «эффективное число вариантов, из которых модель сомневается при выборе следующего токена».

---

# Глава 1. Как устроены современные LLM

## 1.1 Что такое LLM на самом деле

LLM (Large Language Model, большая языковая модель) — это просто очень большая нейросеть, которая умеет отвечать на вопрос «какой токен вероятнее всего идёт следующим?».

Всё волшебство, которое мы наблюдаем у ChatGPT, Claude, Llama, — это следствие того, что модель, обученная **только** на этой задаче на **триллионах** токенов текста, начинает «понимать» язык, логику, факты о мире и даже некоторые формы рассуждения.

**Почему такая простая задача даёт такой сильный результат?** Потому что чтобы хорошо предсказывать следующее слово в тексте, модели приходится выучить:

- **грамматику** («после артикля *the* обычно идёт существительное»);
- **факты** («Париж — столица …»);
- **логику** («если A > B и B > C, то A > …»);
- **стиль** («в коде на Python после `def` идёт имя функции»).

Это и есть «эмерджентное поведение»: сложные способности возникают из простой задачи при большом масштабе.

## 1.2 Иерархия: AI → ML → DL → LLM

Прежде чем говорить о трансформерах, важно зафиксировать, как связаны термины:

- **AI (Artificial Intelligence).** Самое широкое: любая система, которая решает задачи, требующие «человеческого интеллекта».
- **ML (Machine Learning).** Подмножество AI: системы, которые учатся на данных, а не программируются явно.
- **DL (Deep Learning).** Подмножество ML: обучение многослойных нейросетей.
- **LLM.** Частный случай DL-модели, специализированный на тексте.

Когда в новостях говорят «AI», часто имеют в виду именно LLM. В этой книге AI = контекст, ML = способ обучения, DL = архитектура, LLM = конкретный тип DL-модели.

## 1.3 Токены, эмбеддинги, вероятности — минимальный набор

Разберём на конкретном примере, что происходит, когда модель генерирует текст.

**Шаг 1. Токенизация.** Текст `"Привет, мир"` разбивается на токены и заменяется числами: `[15496, 11, 995]`.

**Шаг 2. Эмбеддинги.** Каждое число превращается в вектор. Например, для модели с размерностью 768 токен `15496` превращается в вектор из 768 чисел.

**Шаг 3. Transformer.** Векторы прогоняются через несколько слоёв attention и FFN. На выходе — вектор для каждого токена.

**Шаг 4. LM-head.** Последний вектор преобразуется в распределение вероятностей по всему словарю (например, 50 000 токенов). Для каждого возможного «следующего» токена получаем число — его вероятность.

**Шаг 5. Sampling.** Из распределения выбирается один токен (по правилу top-k, top-p, temperature). Он добавляется к последовательности, и цикл повторяется.

Всё, что мы будем делать в книге, — реализовывать каждый шаг этой цепочки.

## 1.4 Что изменилось с GPT-2

GPT-2 (2019) — отличный ориентир, потому что его архитектура опубликована и веса открыты. Современные модели (Llama 3, Qwen, DeepSeek-V3) — это тот же скелет с шестью заменами:

| Компонент | GPT-2 | Современные LLM |
|---|---|---|
| Нормализация | LayerNorm | **RMSNorm** (дешевле, стабильнее) |
| Позиции | Обучаемые абсолютные | **RoPE** (лучше экстраполяция) |
| FFN | GELU | **SwiGLU** (лучше качество) |
| Attention | MHA (Multi-Head) | **GQA** (меньше KV-cache) |
| FFN тип | Dense | Dense или **MoE** (Mixture-of-Experts) |
| Residual | Pre-norm/Post-norm | **Pre-norm** (стандарт) |

Каждую замену мы реализуем с нуля и объясним, зачем она нужна.

## 1.5 Общая схема современного LLM

Вот что мы построим к концу главы 5:

```text
вход: ID токенов [B, T]
   ↓
Embedding: [B, T, D]                — каждое число превращается в вектор
   ↓
Transformer Block × L:              — L одинаковых блоков
   ├── RMSNorm                       — нормализация
   ├── GQA-Attention с RoPE          — «кто на кого смотрит»
   ├── Residual (+ вход)
   ├── RMSNorm
   ├── SwiGLU FFN                    — «что запомнить»
   └── Residual (+ вход)
   ↓
RMSNorm (финальный)
   ↓
LM-head (Linear D → vocab)          — превращаем вектор в вероятности токенов
   ↓
выход: логиты [B, T, vocab]
```

где `B` — размер батча, `T` — длина последовательности, `D` — размерность эмбеддинга, `L` — число слоёв, `vocab` — размер словаря.

## 1.6 Первый код: RMSNorm

Начнём с самой простой замены — RMSNorm. Сначала разберёмся, что такое нормализация и зачем она нужна.

**Проблема.** В глубокой нейросети значения после каждого слоя могут расти или уменьшаться экспоненциально. Если они станут огромными — градиенты «взорвутся» и обучение остановится. Если крошечными — «затухнут» и модель ничего не выучит. Нормализация — это приведение значений к «здоровому» масштабу перед каждым слоем.

**LayerNorm (старый способ).** Для каждого вектора $x$ считает среднее $\mu$ и стандартное отклонение $\sigma$, затем делает $y = (x - \mu)/\sigma$. После этого применяет обучаемые `scale` и `shift`: $y \cdot \gamma + \beta$. Это отлично работает, но требует двух проходов по вектору.

**RMSNorm (новый способ).** Замечание: вычитание среднего не так важно — можно обойтись без него. Остаётся только нормировка на «корень из среднего квадрата» (Root Mean Square):

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma
$$

где $\epsilon$ — маленькая константа (обычно $10^{-6}$) для защиты от деления на ноль, а $\gamma$ — обучаемый вектор масштаба.

**Почему это лучше?**

1. Меньше вычислений: нет вычитания среднего, нет `shift`.
2. Работает не хуже (эмпирически).
3. Стандарт для всех современных моделей.

### Файл `model/rmsnorm.py`

```python
"""RMSNorm — нормализация по корню из среднего квадрата.

Используется во всех современных LLM (Llama, Qwen, Mistral, DeepSeek).
Отличие от LayerNorm: не вычитаем среднее, не используем shift.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma.

    Параметры
    ----------
    dim : int
        Размерность последнего измерения входного тензора.
    eps : float
        Маленькая константа, чтобы не делить на ноль.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # gamma — обучаемый масштаб. Инициализируем единицами:
        # в начале обучения нормализация «не портит» сигнал.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) или (..., D)
        # rms: (B, T, 1) — корень из среднего квадрата по последней оси.
        # keepdim=True нужен для последующего деления (broadcasting).
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        # rsqrt() — это 1/sqrt(), быстрее и точнее, чем 1/torch.sqrt().
        return x * rms * self.weight
```

### Проверка: `exercises/check_rmsnorm.py`

```python
"""Проверка RMSNorm: после применения средний квадрат должен быть ≈ 1."""

import torch
from model.rmsnorm import RMSNorm


def main():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8) * 10  # нарочно «раздутые» значения
    norm = RMSNorm(dim=8)
    y = norm(x)

    # Считаем средний квадрат по последней оси.
    ms = y.pow(2).mean(dim=-1)
    print("Форма входа:", x.shape)
    print("Форма выхода:", y.shape)
    print("Средний квадрат (должно быть ~1.0):", ms)
    assert torch.allclose(ms, torch.ones_like(ms), atol=1e-3)


if __name__ == "__main__":
    main()
```

Запустите `python -m exercises.check_rmsnorm` — увидите, что все значения ≈ 1.0. Нормализация работает.

## 1.7 Резюме главы

- LLM — это большая нейросеть, обучаемая предсказывать следующий токен.
- Современная архитектура = GPT-2 + RMSNorm, RoPE, SwiGLU, GQA.
- Нормализация стабилизирует обучение; RMSNorm — самый простой и популярный вариант.
- Весь стек можно реализовать на PyTorch в сотни строк.

---

# Глава 2. Токенизация и подготовка данных

## 2.1 Зачем нужна токенизация: подробно

Модель — это математика над числами. Нельзя подать на вход `"кот"`. Нужно превратить текст в последовательность целых чисел. Этим занимается **токенизатор**.

Наивное решение — «одно слово = один ID»:

```python
word2id = {"кот": 1, "сел": 2, "на": 3, "ковёр": 4, ...}
```

Проблемы:

1. **Словарь бесконечен.** В русском сотни тысяч словоформ; в коде — миллионы идентификаторов.
2. **OOV (out-of-vocabulary).** Если модель встретит слово, которого нет в словаре, она не сможет его обработать. А встретит она его обязательно — язык бесконечен.
3. **Плохая компрессия.** Редкие слова занимают целый токен, хотя могли бы занимать 2–3 сабворда.

**Решение — subword-токенизация.** Разбиваем текст на куски, которые не являются целыми словами, но достаточно частотны. Например:

- `"программирование"` → `["программ", "ирование"]`
- `"unbelievable"` → `["un", "believ", "able"]`
- `"GPT-4"` → `["G", "PT", "-", "4"]`

Тогда словарь содержит, например, 50 000 сабвордов, и **любое** слово (даже незнакомое) можно разбить на них.

## 2.2 Как работает BPE: пошагово

BPE (Byte Pair Encoding) — самый популярный алгоритм. Идея проста: берём самые частые пары соседних токенов и склеиваем их в один токен. Повторяем $N$ раз.

**Пример.** Пусть есть текст `"low low low lower lower newest newest newest newest"`. Начальный словарь — все отдельные символы:

```text
l o w _ l o w _ l o w _ l o w e r _ l o w e r _ n e w e s t ...
```

Шаг 1. Считаем пары. Самая частая — `l o` (появляется много раз). Склеиваем `l` + `o` → `lo`.

Шаг 2. Считаем снова. Самая частая — `lo w`. Склеиваем → `low`.

Шаг 3. Самая частая теперь `e s`. Склеиваем → `es`. И так далее.

Через $N$ шагов получаем словарь, где частотные куски — цельные токены, а редкие слова автоматически разбиваются на более мелкие.

**Почему именно байты?** В современных реализациях (tiktoken, SentencePiece) начальный словарь — 256 байтов, а не букв. Это гарантирует, что **любой** UTF-8-текст, включая эмодзи и иероглифы, разбивается на известные токены.

## 2.3 Токенизация на практике с tiktoken

```python
import tiktoken

# Загружаем токенизатор GPT-2 (тот же словарь, что в ChatGPT-3.5/4, только больше).
enc = tiktoken.get_encoding("gpt2")

text = "Привет, мир! Hello, world."
ids = enc.encode(text)
print("ID токенов:", ids)
print("Длина:", len(ids))

# Обратное преобразование.
print("Декод:", enc.decode(ids))
```

### Файл `data/tokenizer_demo.py`

```python
"""Демонстрация токенизации через tiktoken.

Устанавливается: pip install tiktoken
"""

import tiktoken


def demo(text: str = "Привет, мир! Hello, world.") -> None:
    enc = tiktoken.get_encoding("gpt2")

    ids = enc.encode(text)
    print(f"Текст:      {text!r}")
    print(f"Токены ID:  {ids}")
    print(f"Кол-во:     {len(ids)}")
    print(f"Декодинг:   {enc.decode(ids)!r}")

    # Разбираем по токенам, чтобы увидеть разбиение.
    print("\nПо токенам:")
    for i in ids:
        # decode_single_token_bytes даёт «сырые» байты токена.
        piece = enc.decode_single_token_bytes(i)
        try:
            readable = piece.decode("utf-8")
        except UnicodeDecodeError:
            # Если токен — часть многобайтового символа, показываем байты.
            readable = repr(piece)
        print(f"  {i:>7} → {readable!r}")


if __name__ == "__main__":
    demo()
```

Запустите — увидите, как `"Привет"` разбилось на несколько токенов (для GPT-2 русский «дорогой»), а `"Hello"` — на один-два. **Отсюда практический вывод:** русскоязычные модели (Qwen, YandexGPT, Llama с расширенным словарём) используют токенизаторы, оптимизированные под кириллицу.

## 2.4 Датасет для pretraining

**Что такое pretraining-датасет?** Это огромный массив текста. Задача датасета — резать его на куски фиксированной длины и возвращать пары `(x, y)`, где `y` — это `x`, сдвинутый на 1.

Пример. Пусть корпус — `[10, 20, 30, 40, 50, 60]`, длина блока 4. Тогда:

- Пример 1: `x = [10, 20, 30, 40]`, `y = [20, 30, 40, 50]`
- Пример 2: `x = [20, 30, 40, 50]`, `y = [30, 40, 50, 60]`

Модель учится: «увидев `[10, 20, 30, 40]`, предскажи `[20, 30, 40, 50]`». Поскольку обучение идёт параллельно по всем позициям, каждый пример даёт 4 обучающих сигнала сразу.

### Файл `data/dataset.py`

```python
"""Датасет для pretraining (causal language modeling).

Каждый пример — пара (x, y), где y — это x, сдвинутый на один токен.
"""

from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class CausalLMDataset(Dataset):
    """Режет корпус на блоки фиксированной длины.

    Параметры
    ----------
    token_ids : torch.Tensor
        Одномерный тензор ID токенов всего корпуса.
    block_size : int
        Длина блока (контекста) модели.
    stride : Optional[int]
        Шаг окна. Если None, равен block_size (без перекрытия).
        Если < block_size, окна перекрываются — больше примеров,
        но и больше риск переобучения.
    """

    def __init__(
        self,
        token_ids: torch.Tensor,
        block_size: int,
        stride: Optional[int] = None,
    ) -> None:
        self.tokens = token_ids
        self.block_size = block_size
        self.stride = stride or block_size

    def __len__(self) -> int:
        # Сколько блоков помещается в корпус с учётом сдвига.
        # -1 нужен, чтобы target не вылез за границу.
        return max(
            0,
            (len(self.tokens) - self.block_size - 1) // self.stride + 1,
        )

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.block_size
        x = self.tokens[start:end]           # вход
        y = self.tokens[start + 1 : end + 1] # сдвиг на 1
        return x, y


def build_pretrain_loader(
    text: str,
    enc,
    block_size: int = 512,
    batch_size: int = 8,
    stride: Optional[int] = None,
    num_workers: int = 0,
) -> DataLoader:
    """Удобная сборка DataLoader из сырого текста."""
    ids = torch.tensor(enc.encode(text), dtype=torch.long)
    ds = CausalLMDataset(ids, block_size=block_size, stride=stride)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,   # убрать неполный последний батч
        num_workers=num_workers,
    )
```

### Проверка: `exercises/check_dataset.py`

```python
"""Проверка CausalLMDataset: y должен быть x, сдвинутым на 1."""

import torch
from data.dataset import CausalLMDataset


def main():
    tokens = torch.arange(10)  # [0,1,2,...,9]
    ds = CausalLMDataset(tokens, block_size=4, stride=4)
    print(f"Примеров: {len(ds)}")  # ожидаем 2

    for i in range(len(ds)):
        x, y = ds[i]
        print(f"Пример {i}:")
        print(f"  x = {x.tolist()}")
        print(f"  y = {y.tolist()}")
        assert torch.equal(x[1:], y[:-1]), "y должен быть x, сдвинутым на 1"
    print("OK: сдвиг работает правильно.")


if __name__ == "__main__":
    main()
```

## 2.5 Специальные токены и ChatML

Модель должна понимать, где заканчивается один документ и начинается другой, где реплика пользователя, где ответ ассистента. Для этого вводят **специальные токены**:

- `<|endoftext|>` — граница документа в pretraining.
- `<s>`, `</s>` — начало/конец последовательности (Llama).
- `<|im_start|>user`, `<|im_end|>` — роли (ChatML, Qwen).
- `<pad>` — заполнитель для батчей разной длины.

В SFT данные выглядят так:

```text
<|im_start|>system
Ты — помощник.<|im_end|>
<|im_start|>user
Как дела?<|im_end|>
<|im_start|>assistant
Отлично, спасибо!<|im_end|>
```

**Важно:** loss в SFT считается **только на токенах ответа ассистента**. Иначе модель учится предсказывать и вопрос, что бесполезно.

## 2.6 Упражнение: BPE с нуля

**Задача.** Реализовать простой BPE-обучатель: на вход текст, на выход — словарь и функция токенизации.

**Полное решение.** Файл `exercises/ex_2_1_bpe_from_scratch.py`:

```python
"""Упражнение 2.1: BPE с нуля.

Задача: обучить BPE на тексте и токенизировать новые строки.

Запуск:
    python -m exercises.ex_2_1_bpe_from_scratch
"""

from collections import Counter
from typing import Dict, List, Tuple


def get_pair_counts(ids_list: List[List[int]]) -> Counter:
    """Считает, сколько раз каждая пара соседних ID встречается.

    ids_list — список последовательностей ID (для батча).
    """
    pairs = Counter()
    for ids in ids_list:
        for a, b in zip(ids, ids[1:]):
            pairs[(a, b)] += 1
    return pairs


def merge_pair(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Заменяет все вхождения пары в последовательности на новый ID."""
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(
    text: str,
    num_merges: int = 20,
) -> Tuple[Dict[int, bytes], List[Tuple[int, int]]]:
    """Обучает BPE.

    Возвращает
    ----------
    vocab : dict[int, bytes]
        Словарь: ID → байты.
    merges : list[tuple[int, int]]
        Список операций слияния в порядке применения.
    """
    # Начальный словарь — все 256 байтов.
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    # Превращаем текст в байты и разбиваем на отдельные байтовые ID.
    ids = list(text.encode("utf-8"))

    merges: List[Tuple[int, int]] = []
    for _ in range(num_merges):
        if len(ids) < 2:
            break
        counts = get_pair_counts([ids])
        if not counts:
            break
        # Самая частая пара.
        pair, _ = counts.most_common(1)[0]
        merges.append(pair)
        vocab[next_id] = vocab[pair[0]] + vocab[pair[1]]
        ids = merge_pair(ids, pair, next_id)
        next_id += 1
    return vocab, merges


def encode(text: str, vocab: Dict[int, bytes], merges: List[Tuple[int, int]]) -> List[int]:
    """Токенизирует текст с уже обученным BPE."""
    ids = list(text.encode("utf-8"))
    for pair in merges:
        # Для каждого слияния получаем ID нового токена из словаря.
        new_id = None
        for k, v in vocab.items():
            if v == vocab[pair[0]] + vocab[pair[1]]:
                new_id = k
                break
        if new_id is None:
            continue
        ids = merge_pair(ids, pair, new_id)
    return ids


def main() -> None:
    text = "low low low lower lower newest newest newest newest"
    vocab, merges = train_bpe(text, num_merges=10)

    print(f"Слитий: {len(merges)}")
    print(f"Размер словаря: {len(vocab)}")
    print("\nСлияния по порядку:")
    for i, pair in enumerate(merges):
        merged = vocab[pair[0]] + vocab[pair[1]]
        print(f"  {i+1}: {vocab[pair[0]]!r} + {vocab[pair[1]]!r} → {merged!r}")

    print("\nКодируем «lower newest»:")
    ids = encode("lower newest", vocab, merges)
    print("  IDs:", ids)
    print("  Байты:", b"".join(vocab[i] for i in ids))


if __name__ == "__main__":
    main()
```

Обратите внимание: `encode` в решении неоптимален (перебирает словарь для каждого слияния), но зато самый понятный. Упражнение для читателя — сделать быстрый вариант с индексом.

## 2.7 Резюме главы

- Токенизатор превращает текст в ID; BPE — стандарт для LLM.
- Pretraining-датасет делает пары `(x, y)`, где `y` — сдвиг `x` на 1.
- SFT использует специальные токены и loss только на ответе ассистента.
- Упражнение показало, как реализовать BPE с нуля.

---

# Глава 3. Внимание: от dot-product до FlashAttention

## 3.1 Зачем нужно внимание: аналогия с библиотекой

Представьте, что каждый токен — это человек в библиотеке. У него есть **вопрос** (query): «кто из присутствующих относится к местоимению "он"?». У каждого другого токена есть **ярлык на корешке** (key), по которому можно понять, о чём он. И у каждого есть **содержание** (value) — то, что он готов рассказать.

Механизм attention делает следующее:

1. Для каждого токена строит **query**.
2. Для каждого токена строит **key**.
3. Для каждого токена строит **value**.
4. Считает «похожесть» query с каждым key — это **attention scores**.
5. Применяет softmax, чтобы получить веса (сумма = 1).
6. Итог для токена = взвешенная сумма **value** всех токенов.

Так каждый токен «собирает» контекст из всех остальных.

## 3.2 Self-attention математически

Дано: вход $X \in \mathbb{R}^{T \times d}$ (T токенов, d признаков). Обучаемые матрицы $W_Q, W_K, W_V$ размером $d \times d_k$.

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

**Attention scores** — это произведение Q на K-транспонированное:

$$
S = \frac{QK^\top}{\sqrt{d_k}}
$$

Размер $S$ — $T \times T$: для каждой пары (i, j) — «насколько токен i смотрит на токен j».

**Зачем делим на $\sqrt{d_k}$?** Если размерность большая, скалярные произведения получаются большими. Softmax от больших чисел даёт «почти one-hot» распределение (одна вероятность ≈ 1, остальные ≈ 0), и градиенты становятся крошечными. Масштабирование выравнивает распределение.

**Causal-маска.** Для генерации модели нельзя смотреть на будущие токены. Ставим в матрице $S$ на позиции $(i, j)$ при $j > i$ значение $-\infty$, чтобы после softmax там был 0.

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

## 3.3 Наивная реализация

### Файл `model/attention.py` (часть 1)

```python
"""Внимание: наивное, PyTorch SDPA и GQA.

Мы начинаем с наивной реализации — она медленная, но понятная.
Затем заменим её на F.scaled_dot_product_attention, который
внутри использует FlashAttention, если доступен.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Наивное self-attention.

    Параметры
    ----------
    q, k, v : torch.Tensor
        Форма (B, H, T, d) — батч, головы, длина, размерность головы.
    causal : bool
        Если True, токен i не видит токены j > i.

    Возвращает
    ----------
    torch.Tensor
        Форма (B, H, T, d).
    """
    d = q.size(-1)
    # S = (B, H, T, T)
    scores = (q @ k.transpose(-2, -1)) * (d ** -0.5)

    if causal:
        T = q.size(-2)
        # Верхнетреугольная маска из True (для j > i).
        mask = torch.triu(
            torch.ones(T, T, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        # Заполняем -inf там, где mask True.
        scores = scores.masked_fill(mask, float("-inf"))

    # softmax по последней оси (по j).
    weights = scores.softmax(dim=-1)
    return weights @ v
```

**Что тут происходит по шагам:**

1. `q @ k.transpose(-2, -1)` — матричное умножение по последним двум осям. Размер: `(B, H, T, d) @ (B, H, d, T) = (B, H, T, T)`. Это матрица «кто на кого смотрит».
2. `* (d ** -0.5)` — масштабирование.
3. `masked_fill(mask, -inf)` — запрещаем смотреть вперёд.
4. `softmax(dim=-1)` — превращаем в вероятности (сумма по строке = 1).
5. `weights @ v` — взвешенная сумма значений.

## 3.4 PyTorch SDPA: тот же результат, но быстрее

В PyTorch 2.0 появилась функция `F.scaled_dot_product_attention`, которая:

- Выбирает оптимальную реализацию в зависимости от GPU.
- На NVIDIA Ampere+ использует **FlashAttention** — не материализует матрицу $T \times T$ в памяти.
- Даёт тот же математический результат.

### Файл `model/attention.py` (часть 2)

```python
def attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """То же, что attention_naive, но через F.scaled_dot_product_attention.

    На GPU с поддержкой FlashAttention это на порядок быстрее
    при длинных контекстах.
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
```

## 3.5 Что такое FlashAttention на самом деле

**Проблема наивного attention.** Матрица $S$ размером $T \times T$ хранится в памяти GPU (HBM). При $T=8192$ и 32 головах это $8192^2 \times 32 \times 4$ байт ≈ **8.6 ГБ** только на одну матрицу. Для длинных контекстов (128k) это неподъёмно.

**Идея FlashAttention.** Не храним $S$ целиком. Разбиваем $Q, K, V$ на блоки, которые помещаются в SRAM (быстрая память, ~20 МБ). Для каждого блока $Q$ проходим по блокам $K, V$ и **накапливаем** результат, поддерживая «бегущие» максимум и сумму для softmax (online softmax).

**Итог:**

- Память $O(T)$ вместо $O(T^2)$.
- Скорость 2–4× выше при длинном контексте.
- Математически идентично наивному.

Использовать его вручную не нужно — `SDPA` сам выберет оптимальный backend.

## 3.6 GQA: что это и зачем

**MHA (Multi-Head Attention, классика).** 32 головы → 32 независимых Q, K, V. KV-cache при генерации огромен.

**MQA (Multi-Query Attention).** 32 Q-головы, но 1 общая K и 1 общая V. KV-cache в 32 раза меньше, но качество падает.

**GQA (Grouped-Query Attention).** Компромисс: 32 Q-головы разбиваем на 8 групп по 4. Каждая группа делит одну пару K/V. KV-cache в 4 раза меньше, качество почти как у MHA.

Именно GQA используют Llama 2/3, Mistral, Qwen, Gemma.

### Файл `model/attention.py` (часть 3)

```python
class GroupedQueryAttention(nn.Module):
    """GQA-внимание с опциональным RoPE.

    Параметры
    ----------
    d_model : int
        Размерность эмбеддинга (вход и выход).
    n_heads : int
        Число query-голов.
    n_kv_heads : int
        Число KV-голов. Должно делить n_heads.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model должен делиться на n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads должен делиться на n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.repeat = n_heads // n_kv_heads

        # Проекции Q, K, V, O. Без bias — как в современных LLM.
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Проекции и reshape: (B, T, n_heads * head_dim) → (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Расширяем K, V по числу групп: (B, n_kv_heads, T, d) → (B, n_heads, T, d)
        k = k.repeat_interleave(self.repeat, dim=1)
        v = v.repeat_interleave(self.repeat, dim=1)

        # SDPA автоматически выберет FlashAttention, если доступен.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Обратный reshape: (B, n_heads, T, d) → (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(y)
```

## 3.7 Упражнение: сравнить наивное и SDPA

**Задача.** Убедиться, что `attention_naive` и `attention_sdpa` дают одинаковый результат.

**Решение.** Файл `exercises/ex_3_1_attention_naive.py`:

```python
"""Упражнение 3.1: сравнить наивное attention и SDPA.

Запуск:
    python -m exercises.ex_3_1_attention_naive
"""

import torch

from model.attention import attention_naive, attention_sdpa


def main():
    torch.manual_seed(42)
    B, H, T, d = 2, 4, 8, 16

    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, d)
    v = torch.randn(B, H, T, d)

    out_naive = attention_naive(q, k, v, causal=True)
    out_sdpa = attention_sdpa(q, k, v, causal=True)

    print(f"Форма выхода: {out_naive.shape}")
    max_diff = (out_naive - out_sdpa).abs().max().item()
    print(f"Макс. разница: {max_diff:.2e}")
    assert max_diff < 1e-5, "Результаты должны совпадать"
    print("OK: реализации эквивалентны")


if __name__ == "__main__":
    main()
```

Если всё правильно, разница ≈ $10^{-7}$ (это погрешность float32).

## 3.8 Резюме главы

- Attention: «смотрит» друг на друга, собирает контекст.
- Формула: `softmax(QK^T / sqrt(d)) V` с causal-маской.
- SDPA даёт тот же результат, но с FlashAttention — быстрее и экономнее по памяти.
- GQA уменьшает KV-cache в `n_heads / n_kv_heads` раз.

---

# Глава 4. Современный transformer block

## 4.1 Что внутри одного блока

Современный блок (post-norm в оригинальной статье заменён на **pre-norm**):

```text
x = x + Attention(RMSNorm(x))     # attention и residual
x = x + FFN(RMSNorm(x))           # FFN и residual
```

**Зачем два residual?** Residual (пропуск через $+x$) помогает градиентам «перепрыгивать» через слои. Без него в глубокой сети градиент затухает.

**Зачем pre-norm?** Normalize **до** attention, а не после. Это делает обучение стабильнее — можно тренировать сотни слоёв.

## 4.2 RoPE: относительные позиции через вращение

**Проблема.** Attention сам по себе **не знает** позиции. Если поменять местами слова, веса attention будут те же. Нужно «влить» информацию о позиции.

**Плохое решение (абсолютные эмбеддинги).** Прибавить к эмбеддингу токена вектор позиции: `embed[token] + pos_embed[position]`. Проблема: модель не умеет обобщать на позиции длиннее, чем видела при обучении.

**Хорошее решение (RoPE, Rotary Position Embedding).** Не прибавлять, а **вращать** Q и K на угол, зависящий от позиции. Красивое свойство: скалярное произведение $q_m^\top k_n$ зависит только от **разности** $m - n$, а не от абсолютных позиций. Это и есть «относительные позиции».

**Как это работает математически.** Вектор $d$-мерный, разбиваем на пары $(x_{2i}, x_{2i+1})$. Каждую пару вращаем на угол $m \cdot \theta_i$, где $m$ — позиция, $\theta_i = 10000^{-2i/d}$.

$$
\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} =
\begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

**Интуиция.** Представьте, что каждый токен — стрелка на циферблате. RoPE поворачивает стрелку на угол, зависящий от позиции. Два токена «смотрят» друг на друга под углом, зависящим от разницы их позиций — это и есть «относительное расстояние».

### Файл `model/rope.py`

```python
"""RoPE (Rotary Position Embedding).

Используется в Llama, Qwen, Mistral, DeepSeek и почти везде.
"""

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Кэширует cos/sin для позиций от 0 до max_seq_len-1."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "dim должен быть чётным"
        self.dim = dim

        # Частоты: theta_i = base^(-2i/dim), i = 0, 1, ..., dim/2 - 1.
        # Берём каждое второе измерение от 0 до dim-2.
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        # Углы для каждой позиции: (max_seq_len, dim/2).
        t = torch.arange(max_seq_len).float()
        angles = torch.outer(t, freqs)

        # Кэшируем cos и sin как буферы (не параметры).
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Применить RoPE к x: (B, H, T, head_dim)."""
        T = x.shape[-2]
        cos = self.cos_cached[:T].to(x.dtype)  # (T, dim/2)
        sin = self.sin_cached[:T].to(x.dtype)

        # Добавляем оси для broadcasting: (1, 1, T, dim/2).
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Разбиваем на две половины.
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2 :]

        # Поворот:
        # x' = x1 * cos - x2 * sin
        # y' = x1 * sin + x2 * cos
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```

**Обратите внимание:** RoPE применяется **только к Q и K**, но не к V. Почему? Потому что нам важно, чтобы «кто на кого смотрит» зависело от относительных позиций. Само содержимое (V) — не зависит от позиции.

## 4.3 SwiGLU: улучшенный FFN

**Классический FFN** (в GPT-2):

```text
FFN(x) = W2 · GELU(W1 · x)
```

Два линейных слоя, между ними активация.

**SwiGLU** (в Llama, Qwen, Mistral):

```text
SwiGLU(x) = W_down · (Swish(W_gate · x) ⊙ (W_up · x))
```

Три линейных слоя и поэлементное умножение. Слово «gate» означает «ворота»: одна ветка **разрешает** второй ветке передавать сигнал, а не разрешает.

$$
\mathrm{SwiGLU}(x) = \mathrm{Swish}(xW_{\text{gate}}) \odot (xW_{\text{up}}) \, W_{\text{down}}
$$

где $\mathrm{Swish}(z) = z \cdot \sigma(z)$.

**Зачем?** Эмпирически даёт лучшее качество при том же числе параметров (если правильно подобрать `hidden_dim`). Плюс лучше работает с длинными контекстами.

### Файл `model/swiglu.py`

```python
"""SwiGLU FFN — стандарт для современных LLM."""

import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU(x) = W_down · (Swish(W_gate·x) ⊙ (W_up·x)).

    Параметры
    ----------
    d_model : int
        Размерность входа/выхода.
    hidden_mult : float
        Во сколько раз скрытый слой больше d_model.
        В Llama используется 8/3 ≈ 2.67, чтобы скомпенсировать
        три матрицы (gate, up, down) вместо двух у классического FFN.
    """

    def __init__(self, d_model: int, hidden_mult: float = 8 / 3):
        super().__init__()
        hidden = int(hidden_mult * d_model)
        # Округляем до кратного 64 — это ускоряет матричные умножения.
        hidden = (hidden + 63) // 64 * 64
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        # F.silu(z) = z * sigmoid(z) = Swish(z)
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

## 4.4 Собираем transformer block

### Файл `model/block.py`

```python
"""Один transformer-блок современного образца (pre-norm)."""

import torch.nn as nn

from .attention import GroupedQueryAttention
from .rmsnorm import RMSNorm
from .rope import RoPE
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """Блок: RMSNorm → GQA+RoPE → Residual → RMSNorm → SwiGLU → Residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.rope = RoPE(d_model // n_heads, max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x, kv_cache=None):
        # Attention-ветка.
        h = self.norm1(x)
        # RoPE применяется внутри attention к q, k. Для простоты
        # в этом примере RoPE стоит как отдельный модуль — расширим
        # позже при добавлении KV-cache.
        h = self.attn(h)
        x = x + h

        # FFN-ветка.
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x
```

**Замечание.** Для простоты в этой главе RoPE не подключён к attention внутри блока (это сделаем в главе 5). Пока важно понять структуру: `norm → attention → residual → norm → ffn → residual`.

## 4.5 Упражнение: визуализировать RoPE

**Задача.** Построить график: как меняется скалярное произведение двух векторов, закодированных RoPE, в зависимости от расстояния между позициями.

**Решение.** Файл `exercises/ex_4_1_rope_visualize.py`:

```python
"""Упражнение 4.1: визуализация RoPE.

Показывает, что скалярное произведение q·k зависит только
от разности позиций. Запуск:
    python -m exercises.ex_4_1_rope_visualize
"""

import torch
import matplotlib.pyplot as plt

from model.rope import RoPE


def main():
    torch.manual_seed(0)
    dim = 16
    max_seq_len = 64

    rope = RoPE(dim=dim, max_seq_len=max_seq_len)

    # Один фиксированный q в позиции 0.
    # Один фиксированный k, который мы «двигаем» по позициям.
    q = torch.randn(1, 1, 1, dim)  # (B, H, T=1, dim)
    k_base = torch.randn(1, 1, 1, dim)

    # Для каждой позиции m: применяем RoPE к k в этой позиции.
    dots = []
    for m in range(max_seq_len):
        # Применяем RoPE к k, но только к позиции m.
        # Проще: сделаем общий тензор и применим ко всем, затем возьмём m-й.
        k_all = k_base.expand(1, 1, max_seq_len, dim).contiguous()
        k_rotated = rope(k_all)
        # Теперь q в позиции 0, k в позиции m.
        # Чтобы q тоже был «в позиции 0» (нулевое вращение), просто
        # берём q как есть. Скалярное произведение:
        dot = (q * k_rotated[:, :, m : m + 1, :]).sum().item()
        dots.append(dot)

    plt.figure(figsize=(8, 4))
    plt.plot(range(max_seq_len), dots)
    plt.xlabel("Позиция k (q в позиции 0)")
    plt.ylabel("q · k после RoPE")
    plt.title("RoPE: скалярное произведение зависит от расстояния")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rope_visualization.png", dpi=150)
    print("Сохранено в rope_visualization.png")


if __name__ == "__main__":
    main()
```

Если всё правильно, график покажет не строго убывающую, но затухающую (в среднем) кривую — чем дальше позиция, тем меньше корреляция.

## 4.6 Резюме главы

- Блок = pre-norm + attention + residual + pre-norm + FFN + residual.
- RoPE кодирует позиции через вращение, работает для относительных расстояний.
- SwiGLU — улучшенный FFN, стандарт для Llama/Qwen/Mistral.
- Все три компонента реализованы и проверены.

---

# Глава 5. Сборка LLM и pretraining

## 5.1 Полная модель

Теперь соединим всё в одну модель. Структура:

```text
вход: ID токенов [B, T]
   ↓
Embedding [B, T, D]
   ↓
TransformerBlock × L
   ↓
RMSNorm (финальный)
   ↓
Linear (LM-head): D → vocab
   ↓
логиты [B, T, vocab]
```

### Файл `config.py`

```python
"""Параметры модели. Меняя их, вы получаете модели разного размера."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50304    # словарь; 50304 делится на 128 — ускоряет GPU
    context_length: int = 1024 # максимальная длина последовательности
    emb_dim: int = 768         # размерность эмбеддинга
    n_heads: int = 12          # query-голов
    n_kv_heads: int = 4        # KV-голов (GQA)
    n_layers: int = 12         # число блоков
    dropout: float = 0.0       # для маленьких данных можно 0.1
```

### Файл `model/gpt.py`

```python
"""Полная модель GPT-подобной LLM."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig
from .block import TransformerBlock
from .rmsnorm import RMSNorm


class MiniGPT(nn.Module):
    """Минимальная современная LLM (GPT-архитектура с RMSNorm/RoPE/SwiGLU/GQA)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        # Эмбеддинги токенов.
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)

        # Стек блоков. nn.ModuleList — чтобы параметры регистрировались.
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=cfg.emb_dim,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                max_seq_len=cfg.context_length,
            )
            for _ in range(cfg.n_layers)
        ])

        # Финальная нормализация и голова.
        self.norm_f = RMSNorm(cfg.emb_dim)
        self.lm_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        # Weight tying: LM-head использует ту же матрицу, что и эмбеддинги.
        # Экономит ~30% параметров и обычно улучшает качество.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Инициализация: N(0, 0.02) как в GPT-2."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Forward pass.

        Параметры
        ----------
        idx : torch.Tensor (B, T)
            ID токенов.
        targets : torch.Tensor (B, T) | None
            Целевые ID токенов. Если переданы, возвращаем ещё и loss.

        Возвращает
        ----------
        logits, loss
        """
        x = self.tok_emb(idx)  # (B, T, D)

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            # Считаем cross-entropy по всем позициям.
            # view(-1, vocab) — «разложить» в (B*T, vocab).
            # ignore_index=-100 — игнорировать padding.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Генерация текста (greedy / top-k / top-p)."""
        for _ in range(max_new_tokens):
            # Обрезаем контекст до максимума.
            idx_cond = idx[:, -self.cfg.context_length :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # берём последнюю позицию

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = sorted_logits.softmax(-1)
                cum = probs.cumsum(-1)
                mask = cum - probs > top_p
                sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_idx, sorted_logits
                )

            probs = logits.softmax(-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
```

## 5.2 Как работает `lm_head` и weight tying

`lm_head` — это `Linear(D, vocab)`. Он превращает $D$-мерный вектор в $vocab$-мерный (логиты по каждому токену).

**Weight tying.** В `lm_head` матрица весов имеет размер $vocab \times D$. В `tok_emb` — тоже $vocab \times D$. Логично использовать **одну и ту же матрицу**: эмбеддинг токена и его «выходная проекция» — это, по сути, одно и то же отображение.

Мы делаем это строкой:

```python
self.lm_head.weight = self.tok_emb.weight
```

Результат: экономия $vocab \times D$ параметров (для $vocab=50k$, $D=768$ это **38 млн** параметров).

## 5.3 Полный цикл pretraining

### Файл `train/schedule.py`

```python
"""Learning rate schedule: linear warmup + cosine decay."""

import math


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    """LR на шаге step.

    Первые `warmup` шагов LR линейно растёт от 0 до base_lr.
    Затем косинусно убывает до 10% от base_lr.
    """
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)

    if step >= total:
        return 0.1 * base_lr

    progress = (step - warmup) / (total - warmup)
    # 0.1 + 0.9 * 0.5 * (1 + cos(pi * progress))
    # при progress=0 → base_lr, при progress=1 → 0.1 * base_lr
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * progress))
```

### Файл `train/train_pretrain.py`

```python
"""Полный скрипт pretraining на маленьком корпусе.

Запуск:
    python -m train.train_pretrain
"""

import time

import torch
import tiktoken

from config import GPTConfig
from data.dataset import build_pretrain_loader
from model.gpt import MiniGPT
from .schedule import lr_at


def get_batch(loader, device):
    x, y = next(iter(loader))
    return x.to(device), y.to(device)


def train(
    model: MiniGPT,
    loader,
    device: str,
    total_steps: int,
    base_lr: float = 3e-4,
    warmup: int = 100,
    log_every: int = 50,
) -> None:
    """Обучение с warmup + cosine."""
    model.to(device)
    # AdamW — стандарт для LLM. weight_decay=0.1 — проверенное значение.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=0.1, betas=(0.9, 0.95)
    )

    model.train()
    t0 = time.time()
    loader_iter = iter(loader)

    for step in range(total_steps):
        # Меняем LR согласно schedule.
        lr = lr_at(step, base_lr, warmup, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Берём батч.
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        x, y = x.to(device), y.to(device)

        # Forward-backward-step.
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping: предотвращаем взрывные градиенты.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0:
            dt = time.time() - t0
            print(f"step {step:5d}/{total_steps} | lr {lr:.2e} | "
                  f"loss {loss.item():.4f} | {dt:.1f}s")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Маленький корпус. Замените на свой файл.
    text = (
        "Жил-был программист. Он любил чистый код и хорошие тесты. "
        "Каждый день он открывал редактор и писал функции. "
    ) * 500

    enc = tiktoken.get_encoding("gpt2")
    loader = build_pretrain_loader(
        text, enc, block_size=256, batch_size=8, stride=128, num_workers=0
    )

    cfg = GPTConfig(
        vocab_size=enc.n_vocab,
        context_length=256,
        emb_dim=256,
        n_heads=8,
        n_kv_heads=2,
        n_layers=6,
    )
    model = MiniGPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {n_params/1e6:.2f}M")

    train(model, loader, device, total_steps=500, base_lr=3e-4, warmup=50)


if __name__ == "__main__":
    main()
```

## 5.4 Упражнение: убедиться, что weight tying работает

**Задача.** Проверить, что после создания модели `model.lm_head.weight` и `model.tok_emb.weight` — один и тот же объект.

**Решение.** Файл `exercises/ex_5_1_weight_tying.py`:

```python
"""Упражнение 5.1: проверить weight tying.

Запуск:
    python -m exercises.ex_5_1_weight_tying
"""

import torch

from config import GPTConfig
from model.gpt import MiniGPT


def main():
    cfg = GPTConfig(
        vocab_size=1000,
        context_length=64,
        emb_dim=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
    )
    model = MiniGPT(cfg)

    # Проверка: веса — тот же объект в памяти.
    assert model.lm_head.weight is model.tok_emb.weight, \
        "Weight tying не применился"

    # Более того, любые изменения весов отразятся на обеих «точках входа».
    with torch.no_grad():
        model.tok_emb.weight[0, 0] = 42.0
    assert model.lm_head.weight[0, 0] == 42.0

    # Считаем параметры.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров с weight tying: {n_params:,}")

    # Ручная оценка без weight tying: добавление vocab*D.
    n_without_tying = n_params + cfg.vocab_size * cfg.emb_dim
    print(f"Параметров без weight tying: {n_without_tying:,}")
    print(f"Экономия: {cfg.vocab_size * cfg.emb_dim:,} параметров")


if __name__ == "__main__":
    main()
```

## 5.5 Резюме главы

- Модель = эмбеддинги + стек блоков + финальная нормализация + LM-head.
- Weight tying экономит память.
- Pretraining = next-token cross-entropy + AdamW + cosine schedule + grad clipping.
- Всё это реализовано и проверено.

---

# Глава 6. Эффективное обучение

## 6.1 Mixed precision (bf16)

**Проблема.** float32 занимает 4 байта, и на обучение 7B-модели уходит много памяти.

**Решение.** bf16 (bfloat16) — 2 байта, но с тем же диапазоном значений, что у fp32.

```python
from torch.amp import autocast

with autocast(device_type="cuda", dtype=torch.bfloat16):
    logits, loss = model(x, y)
loss.backward()
```

При bf16 scaler не нужен (в отличие от fp16).

### Файл `exercises/ex_6_1_bf16.py`

```python
"""Упражнение 6.1: сравнить скорость fp32 и bf16.

Запуск:
    python -m exercises.ex_6_1_bf16
"""

import time

import torch

from config import GPTConfig
from model.gpt import MiniGPT


def bench(device: str, dtype: torch.dtype, n_steps: int = 20) -> float:
    torch.manual_seed(0)
    cfg = GPTConfig(
        vocab_size=1024, context_length=256, emb_dim=256,
        n_heads=8, n_kv_heads=2, n_layers=4,
    )
    model = MiniGPT(cfg).to(device)
    x = torch.randint(0, 1024, (8, 256), device=device)
    y = torch.randint(0, 1024, (8, 256), device=device)

    # Warmup.
    for _ in range(3):
        with torch.autocast(device_type=device, dtype=dtype):
            _, loss = model(x, y)
        loss.backward()

    t0 = time.time()
    for _ in range(n_steps):
        with torch.autocast(device_type=device, dtype=dtype):
            _, loss = model(x, y)
        loss.backward()
    return (time.time() - t0) / n_steps


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    t32 = bench(device, torch.float32)
    t16 = bench(device, torch.bfloat16)
    print(f"fp32: {t32*1000:.1f} ms/step")
    print(f"bf16: {t16*1000:.1f} ms/step")
    print(f"Ускорение: {t32/t16:.2f}×")


if __name__ == "__main__":
    main()
```

## 6.2 Gradient checkpointing

**Проблема.** При обратном проходе нужно хранить активации каждого слоя. Для 100-слойной модели — огромные тензоры.

**Решение.** Не хранить активации, а пересчитывать их на backward. Трейд-офф: память экономится, но шаг становится на ~20% медленнее.

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def forward(self, x):
        return checkpoint(self._inner, x, use_reentrant=False)

    def _inner(self, x):
        return self.block(x)
```

## 6.3 torch.compile

**Идея.** JIT-компиляция модели в одну оптимизированную функцию. PyTorch делает fusion операций (например, объединяет Linear + GELU).

```python
model = torch.compile(model, mode="max-autotune")
```

Даёт 10–40% ускорения на NVIDIA. Первый шаг компиляции медленный (10–60 сек), потом — быстро.

## 6.4 FSDP: обучение больших моделей

**Проблема.** Модель на 7B в bf16 занимает 14 ГБ только на веса, плюс оптимизатор AdamW (2× fp32 копии) → ~70 ГБ. Одна A100 (80 ГБ) — впритык, а 70B не влезает вообще.

**Решение.** FSDP (Fully Sharded Data Parallel): параметры, градиенты и состояния оптимизатора **шардируются** между GPU. На forward каждый GPU берёт нужный шард, на backward — тоже, и после шага синхронизирует обновления.

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model)
```

Это аналог ZeRO-3 в DeepSpeed. Работает на 8+ GPU.

## 6.5 Резюме главы

- bf16 — почти бесплатное ускорение и экономия памяти.
- Gradient checkpointing — способ влезть в память.
- torch.compile — ускорение без изменения кода.
- FSDP — для моделей, не влезающих в один GPU.

---

# Глава 7. SFT, LoRA/QLoRA и выравнивание

## 7.1 SFT: что это на самом деле

**Supervised Fine-Tuning** — дообучение на парах «инструкция → ответ». Формат:

```text
<|user|>Как приготовить чай?<|assistant|>Вскипятите воду...<|end|>
```

**Ключевое отличие от pretraining:** loss считается **только по токенам ассистента**. Токены пользователя в loss не входят (или входят с меньшим весом).

Зачем? Иначе модель учится предсказывать и вопрос, что бесполезно — вопрос известен заранее.

### Файл `finetune/sft.py`

```python
"""SFT: full fine-tuning на парах (инструкция, ответ)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SFTDataset(torch.utils.data.Dataset):
    """Датасет пар (input_ids, labels, assistant_mask)."""

    def __init__(self, examples, tokenizer, max_len=512):
        self.items = []
        for prompt, response in examples:
            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response) + [tokenizer.eot_token]
            ids = prompt_ids + response_ids
            # -100 — маска для loss (не учитываем в cross-entropy).
            labels = [-100] * len(prompt_ids) + response_ids
            # Обрезаем до max_len.
            self.items.append((
                torch.tensor(ids[:max_len]),
                torch.tensor(labels[:max_len]),
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch, pad_token_id=0):
    """Дополняет батч до максимальной длины в нём."""
    max_len = max(x.size(0) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = max_len - x.size(0)
        xs.append(F.pad(x, (0, pad), value=pad_token_id))
        ys.append(F.pad(y, (0, pad), value=-100))
    return torch.stack(xs), torch.stack(ys)


def sft_loss(logits, labels):
    """Cross-entropy с игнорированием -100 (padding и prompt)."""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )


def train_sft(model, dataset, tokenizer, device, epochs=3, lr=5e-5, batch_size=4):
    model = model.to(device)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = sft_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch+1} loss {loss.item():.4f}")
```

## 7.2 LoRA: дообучение без дообучения

**Проблема.** Полный fine-tuning 7B-модели требует ~120 ГБ GPU. Многие не могут себе это позволить.

**Идея LoRA.** Не трогаем исходные веса $W$. Добавляем две маленькие матрицы $A$ и $B$:

$$
W' = W + \frac{\alpha}{r} BA
$$

где $A \in \mathbb{R}^{r \times d}$, $B \in \mathbb{R}^{d \times r}$, $r \ll d$.

При $r = 8$ или $16$ и $d = 4096$: обучаем $2 \cdot 16 \cdot 4096 \approx 131000$ параметров вместо $16$ миллионов. Память падает в сотни раз.

**Почему работает?** Обновление весов при fine-tuning обычно имеет **низкий ранг** (это эмпирическое наблюдение). Значит, его можно представить как $BA$.

### Файл `finetune/lora.py`

```python
"""LoRA: низкоранговое дообучение."""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Заменяет nn.Linear на тот же Linear + LoRA-обновление.

    Исходные веса заморожены. Обучаются только A и B.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        d_in, d_out = base.in_features, base.out_features
        # A: (r, d_in), B: (d_out, r)
        self.lora_A = nn.Parameter(torch.zeros(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        # Стандартная инициализация: A — гауссово, B — нули.
        # Так на старте W' = W (вклад LoRA нулевой).
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scale = alpha / r

    def forward(self, x):
        # base(x) + scale * (x @ A^T @ B^T)
        return self.base(x) + self.scale * (x @ self.lora_A.T @ self.lora_B.T)


def apply_lora(model: nn.Module, target_names=("q_proj", "v_proj"), r=8, alpha=16):
    """Заменяет указанные Linear на LoRALinear."""
    for name, module in model.named_modules():
        for target in target_names:
            if name.endswith(target):
                # Заменяем модуль.
                parent_name = ".".join(name.split(".")[:-1])
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, name.split(".")[-1], LoRALinear(module, r, alpha))
    return model


def count_trainable(model):
    """Считает обучаемые и замороженные параметры."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
```

## 7.3 DPO: выравнивание без RL

**Проблема.** После SFT модель умеет отвечать, но её ответы не всегда **предпочтительны**. Например, она может быть слишком многословной или грубой.

**DPO** (Direct Preference Optimization) учит модель выбирать «хорошие» ответы из пар «предпочтительный / отвергнутый»:

$$
\mathcal{L}_{\mathrm{DPO}} = -\log\sigma\!\left( \beta \left[ \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right] \right)
$$

где `ref` — SFT-модель (заморожена), $\pi_\theta$ — обучаемая, $\beta$ — параметр «насколько сильно двигаться от ref».

**Интуиция.** Мы хотим, чтобы обучаемая модель **увеличила** вероятность хорошего ответа и **уменьшила** — плохого, но не слишком сильно отклоняясь от ref (иначе модель «сломается»).

### Файл `finetune/dpo.py`

```python
"""DPO — выравнивание по предпочтениям без RL."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dpo_loss(
    logp_chosen_theta: torch.Tensor,
    logp_rejected_theta: torch.Tensor,
    logp_chosen_ref: torch.Tensor,
    logp_rejected_ref: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """DPO-loss.

    Все аргументы — суммы log-вероятностей по токенам ответа.

    При pi_theta == pi_ref loss = log(2), что полезно для проверки.
    """
    chosen_rewards = beta * (logp_chosen_theta - logp_chosen_ref)
    rejected_rewards = beta * (logp_rejected_theta - logp_rejected_ref)
    # DPO-loss = -log sigmoid(chosen - rejected)
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def sequence_logp(model, input_ids, labels):
    """Считает сумму log-вероятностей токенов ответа."""
    logits, _ = model(input_ids)                       # (B, T, vocab)
    log_probs = F.log_softmax(logits, dim=-1)          # (B, T, vocab)
    # Собираем log-prob для правильного токена в каждой позиции.
    gathered = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    # Маскируем -100 (padding/prompt).
    mask = (labels != -100).float()
    return (gathered * mask).sum(dim=-1)               # (B,)


def train_dpo_step(
    model, ref_model, batch, optimizer, beta=0.1,
):
    """Один шаг DPO."""
    chosen_ids, chosen_labels = batch["chosen"]
    rejected_ids, rejected_labels = batch["rejected"]

    logp_c = sequence_logp(model, chosen_ids, chosen_labels)
    logp_r = sequence_logp(model, rejected_ids, rejected_labels)

    with torch.no_grad():
        ref_c = sequence_logp(ref_model, chosen_ids, chosen_labels)
        ref_r = sequence_logp(ref_model, rejected_ids, rejected_labels)

    loss = dpo_loss(logp_c, logp_r, ref_c, ref_r, beta=beta)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

## 7.4 Резюме главы

- SFT учит следовать инструкциям, loss — только по ответу.
- LoRA — дообучение без обновления исходных весов, память падает в сотни раз.
- DPO — выравнивание по предпочтениям без RL-инфраструктуры.

---

# Глава 8. Инференс: KV-cache, квантизация

## 8.1 KV-cache: как ускорить генерацию в разы

**Проблема наивной генерации.** На каждом шаге мы заново считаем attention для всей последовательности. Если длина $T$, то $O(T^2)$ операций на весь текст.

**Решение — KV-cache.** K и V для уже сгенерированных токенов не меняются. Сохраняем их в памяти. На новом шаге считаем Q для нового токена, K и V для него же, добавляем к кэшу — и attention по всему кэшу.

Сложность шага становится $O(T)$ вместо $O(T^2)$.

### Файл `inference/kv_cache.py`

```python
"""Простой KV-cache для инференса."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """Хранит K и V всех уже обработанных токенов.

    Память: 2 × B × H × T × d × bytes_per_element.
    Для Llama 3 8B (H=8, d=128, T=8192, bf16) это ~1 GB.
    """

    def __init__(self, batch_size, n_heads, head_dim, max_len, device, dtype):
        self.k = torch.zeros(
            batch_size, n_heads, max_len, head_dim, device=device, dtype=dtype,
        )
        self.v = torch.zeros_like(self.k)
        self.len = 0  # сколько токенов уже в кэше

    def append(self, k_new, v_new):
        """Добавляет новые K, V: (B, H, 1, d)."""
        T_new = k_new.size(2)
        self.k[:, :, self.len : self.len + T_new] = k_new
        self.v[:, :, self.len : self.len + T_new] = v_new
        self.len += T_new

    def get(self):
        """Возвращает текущие K, V до self.len."""
        return self.k[:, :, : self.len], self.v[:, :, : self.len]


class CachedAttention(nn.Module):
    """Attention с поддержкой KV-cache."""

    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.repeat = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache: KVCache | None = None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if cache is not None:
            cache.append(k, v)
            k, v = cache.get()

        # GQA: повторяем K, V.
        k = k.repeat_interleave(self.repeat, dim=1)
        v = v.repeat_interleave(self.repeat, dim=1)

        # При генерации T=1, query смотрит на все прошлые токены.
        is_causal = (T > 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)
```

## 8.2 Квантизация: 4-бит вместо 16

**Проблема.** Веса bf16 занимают 2 байта на параметр. Для 70B-модели — 140 ГБ. Не влезает в одну GPU.

**Решение.** Квантизация — представление весов меньшим числом бит. 4-битное квантование уменьшает память в 4 раза (35 ГБ для 70B).

**Методы:**

- **GPTQ, AWQ** — post-training, взвешенное квантование с калибровкой.
- **NF4** (bitsandbytes) — для QLoRA.

**Важно:** активации обычно оставляют в bf16, квантуют только веса.

```python
# Пример с bitsandbytes
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", quantization_config=bnb)
```

## 8.3 Упражнение: убедиться, что KV-cache не меняет результат

**Задача.** Проверить, что генерация с KV-cache даёт тот же результат, что без.

**Решение.** Файл `exercises/ex_8_1_kv_cache.py`:

```python
"""Упражнение 8.1: проверить, что KV-cache не меняет результат.

Запуск:
    python -m exercises.ex_8_1_kv_cache
"""

import torch

from inference.kv_cache import CachedAttention, KVCache


def main():
    torch.manual_seed(0)
    d_model, n_heads, n_kv_heads = 32, 4, 2
    attn = CachedAttention(d_model, n_heads, n_kv_heads).eval()
    B, T = 2, 5

    x = torch.randn(B, T, d_model)

    # Без кэша.
    with torch.no_grad():
        y_full = attn(x, cache=None)

    # С кэшем: сначала все токены, потом то же самое.
    with torch.no_grad():
        cache = KVCache(B, n_heads, attn.head_dim, max_len=T, device="cpu", dtype=torch.float32)
        y_cached = attn(x, cache=cache)

    diff = (y_full - y_cached).abs().max().item()
    print(f"Макс. разница: {diff:.2e}")
    assert diff < 1e-5, "KV-cache не должен менять результат"
    print("OK: KV-cache корректен")


if __name__ == "__main__":
    main()
```

## 8.4 Speculative decoding (кратко)

Идея: маленькая draft-модель генерирует K токенов. Большая модель проверяет их за **один** forward. Совпавшие принимаются, остальные пересчитываются.

Ускорение 1.5–3× без потери качества. Требует одинакового токенизатора у обеих моделей.

## 8.5 Резюме главы

- KV-cache — ключевая оптимизация инференса, шаг из $O(T^2)$ становится $O(T)$.
- Квантизация — способ влезть в память, теряя 1–3% качества.
- Speculative decoding — быстрое ускорение без потери качества.
- Всё реализовано и проверено.

---

# Приложение A. Шпаргалка по PyTorch

## A.1 Основные операции

```python
import torch

x = torch.randn(2, 3, 4)         # (B=2, T=3, D=4)
x.shape                          # torch.Size([2, 3, 4])
x.transpose(1, 2).shape          # torch.Size([2, 4, 3])
x.reshape(6, 4).shape            # torch.Size([6, 4])
x.view(-1).shape                 # torch.Size([24])
x @ x.transpose(-2, -1)          # матричное умножение
x.softmax(dim=-1)                # softmax по последней оси
x.log_softmax(dim=-1)            # log(softmax), стабильнее
torch.cat([x, x], dim=1)         # конкатенация
torch.stack([x, x], dim=0)       # новая ось
```

## A.2 Autograd

```python
w = torch.tensor([2.0], requires_grad=True)
loss = (3 * w - 6).pow(2).sum()
loss.backward()                          # считает градиент
print(w.grad)                            # d loss / d w
```

`torch.no_grad()` — отключает построение графа (для инференса).

## A.3 nn.Module

```python
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)
```

`.parameters()` — все обучаемые параметры. `.to(device)` — перенос на GPU.

## A.4 DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class MyDS(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

loader = DataLoader(MyDS([1, 2, 3]), batch_size=2, shuffle=True)
```

## A.5 GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
```

Apple Silicon: `torch.device("mps")`.

---

# Приложение B. Разбор современных архитектур

## B.1 Llama 3

- Размеры: 8B, 70B (и 400B+ в планах).
- RMSNorm (pre-norm).
- RoPE с `base=500000` (для длины 128k).
- GQA: 32 Q-головы, 8 KV-голов для 8B; 64 Q, 8 KV для 70B.
- SwiGLU с `hidden_mult=8/3`.
- Токенизатор: BPE, словарь ~128k.

## B.2 Qwen 2.5 / 3

- От 0.5B до 235B (MoE для больших).
- GQA + RoPE с YaRN для 128k–1M контекста.
- ChatML-шаблон, отличная мультиязычность (включая русский).
- Qwen3 добавляет thinking-режим.

## B.3 DeepSeek-V3

- MoE: 671B всего, 37B активных.
- MLA (Multi-head Latent Attention) — низкоранговое сжатие K, V.
- FP8 обучение.
- GRPO для reasoning (в DeepSeek-R1).

## B.4 Gemma 2 / 3

- RMSNorm, GeGLU (не SwiGLU!), GQA.
- Sliding window attention (локальный + глобальный).
- Дистилляция из больших моделей.

---

# Приложение C. Что делать дальше

**После этой книги:**

1. **Обучите модель на своём корпусе.** Возьмите тексты на русском, обучите токенизатор (SentencePiece), затем pretraining.
2. **Загрузите открытые веса.** Llama, Qwen, Mistral — все на HuggingFace. Совместите с вашим кодом, сверив имена параметров.
3. **Попробуйте LoRA на реальной задаче.** Например, генерация SQL по описанию.
4. **Поиграйте с DPO.** Соберите пары «хороший / плохой ответ» на маленьком датасете.
5. **Разверните инференс.** vLLM + квантизация GPTQ — быстрый сервер на одной GPU.

**Полезные ссылки:**

- Оригинальный transformer: arxiv.org/abs/1706.03762
- GPT-2 paper: openai.com/research/better-language-models
- Llama 3: ai.meta.com/llama
- FlashAttention: arxiv.org/abs/2205.14135
- LoRA: arxiv.org/abs/2106.09685
- DPO: arxiv.org/abs/2305.18290

---

**Конец книги.**

Всё, что описано, реализовано в виде файлов с комментариями. Запускайте, экспериментируйте, ломайте и починяйте — это лучший способ понять, как устроены современные LLM.