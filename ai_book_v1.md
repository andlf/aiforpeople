Современный LLM с нуля: от attention до Chat-модели
Практическое руководство по Python и PyTorch с учётом архитектур 2024–2026 годов

Предисловие
Эта книга — открытое практическое руководство по построению современных больших языковых моделей (LLM) с нуля на Python и PyTorch. Она написана для тех, кто знает Python и базовую математику, но хочет понять, как именно работают модели вроде Llama 3, Qwen 2.5/3, Mistral, Gemma и DeepSeek-V3 — не на уровне «вызвать API», а на уровне «объяснить каждую строку кода».

Структура книги наследует классическому пути «данные → attention → transformer → pretraining → finetuning», но содержание обновлено до состояния 2026 года:

Архитектура: RMSNorm, RoPE, SwiGLU, GQA, MLA, MoE.

Внимание: наивное, PyTorch SDPA и FlashAttention-2/3.

Обучение: mixed precision, gradient checkpointing, torch.compile, FSDP.

Адаптация: LoRA, QLoRA, SFT, DPO, GRPO.

Инференс: KV-cache, квантизация, speculative decoding.

Каждая глава заканчивается рабочим кодом, который можно запустить на одном GPU или в уменьшенном масштабе на CPU.

Оглавление

Как устроены современные LLM

Токенизация и подготовка данных

Внимание: от dot-product до FlashAttention

Современный transformer block: RMSNorm, RoPE, SwiGLU, GQA

Сборка LLM и pretraining

Эффективное обучение: mixed precision, FSDP, torch.compile

SFT, LoRA/QLoRA и выравнивание: DPO/GRPO

Инференс: KV-cache, квантизация, speculative decoding

Приложение A. PyTorch для LLM

Приложение B. Разбор архитектур: Llama 3, Qwen, DeepSeek-V3, Gemma

Приложение C. Упражнения

Глава 1. Как устроены современные LLM
1.1 Что такое LLM
Большая языковая модель — это глубокая нейросеть, обученная предсказывать следующий токен в последовательности. Обученная на триллионах токенов, такая модель приобретает способность понимать инструкции, переводить, суммировать и писать код. В основе почти всех современных LLM лежит transformer — архитектура из статьи Vaswani et al. «Attention Is All You Need» (2017).

Современные LLM отличаются от первых transformer-моделей не принципиальной схемой, а набором инженерных решений, которые повышают качество и снижают стоимость обучения и инференса. Именно эти решения мы будем реализовывать с нуля.

1.2 Что изменилось с GPT-2
Классическая схема «embedding → transformer block × L → final norm → LM head» осталась. Современные открытые модели — это тот же скелет с шестью ключевыми заменами:

Компонент	GPT-2 (2019)	Современные LLM (2024–2026)
Нормализация	LayerNorm	RMSNorm
Позиции	Обучаемые абсолютные	RoPE (+ YaRN для длинного контекста)
Активация FFN	GELU	SwiGLU (иногда GeGLU)
Внимание	MHA	GQA, реже MLA или MQA
FFN	Dense	Dense или Mixture-of-Experts
Residual	Pre-norm / Post-norm	Pre-norm практически везде
Эти изменения дают лучший баланс качества, скорости и памяти. Например, Llama 3 8B использует GQA с 32 query-головами и 8 KV-головами: KV-cache при инференсе сжимается в 4 раза без существенной потери качества.

1.3 Две стадии обучения
Любая современная LLM проходит как минимум две стадии:

Pretraining. Модель обучается предсказывать следующий токен на огромном корпусе неразмеченного текста. На этой стадии формируется «база»: знание языка, мира и некоторых рассуждений. Обучение использует self-supervised loss (кросс-энтропию между предсказанием и истинным следующим токеном).

Post-training. Модель дообучается на меньших, но более качественных данных: инструкциях (SFT), предпочтениях (DPO/GRPO) или наградах (RLHF/RLVR). Здесь формируются умения следовать инструкциям, отказываться от вредных запросов, использовать инструменты.

1.4 Почему современный стек именно такой
RMSNorm вместо LayerNorm
LayerNorm центрирует вектор (вычитает среднее), нормирует по стандартному отклонению и применяет обучаемые 
γ
,
β
γ,β. RMSNorm убирает вычитание среднего:

R
M
S
N
o
r
m
(
x
)
=
x
1
d
∑
i
=
1
d
x
i
2
+
ϵ
⊙
γ
RMSNorm(x)= 
d
1
​
 ∑ 
i=1
d
​
 x 
i
2
​
 +ϵ
​
 
x
​
 ⊙γ
Это дешевле по вычислениям, стабильно в глубоких сетях и стало дефолтом в Llama, Qwen, Mistral, DeepSeek и Gemma.

RoPE вместо абсолютных позиций
Абсолютные обучаемые позиционные эмбеддинги плохо обобщают на длины, не виденные при обучении. RoPE вращает пары компонент query и key на угол, пропорциональный позиции, так что скалярное произведение 
q
m
⊤
k
n
q 
m
⊤
​
 k 
n
​
  зависит только от относительного сдвига 
m
−
n
m−n. Это даёт лучшую экстраполяцию и естественно сочетается с FlashAttention.

SwiGLU вместо GELU
SwiGLU — это gated-FFN:

S
w
i
G
L
U
(
x
)
=
S
w
i
s
h
(
x
W
gate
)
⊙
(
x
W
up
)
 
W
down
SwiGLU(x)=Swish(xW 
gate
​
 )⊙(xW 
up
​
 )W 
down
​
 
При том же числе параметров SwiGLU обычно даёт лучшее качество, чем GELU-FFN, и стал дефолтом для Llama, Qwen, Mistral, DeepSeek.

GQA и MLA вместо MHA
MHA хранит отдельные K и V для каждой query-головы, что раздувает KV-cache при инференсе. GQA группирует query-головы и делит одну пару K/V на группу: KV-cache уменьшается в 
G
G раз. MLA (DeepSeek-V2/V3) сжимает K/V в низкоранговое латентное представление и восстанавливает их на лету, что радикально снижает стоимость длинного контекста.

1.5 Минимальный современный causal LM на PyTorch
Скелет, показывающий, где вставляются современные компоненты. RMSNorm и GQA уже на месте; RoPE и SwiGLU добавим в главе 4.

python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModernConfig:
    vocab_size: int = 32000
    context_length: int = 4096
    emb_dim: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4       # GQA: 12 query heads -> 4 KV heads
    n_layers: int = 12
    dropout: float = 0.0

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

class CausalSelfAttention(nn.Module):
    """Упрощённое GQA-внимание; RoPE добавим в главе 4."""
    def __init__(self, cfg: ModernConfig):
        super().__init__()
        assert cfg.emb_dim % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.emb_dim // cfg.n_heads
        self.q_proj = nn.Linear(cfg.emb_dim, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.emb_dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.emb_dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        repeat = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        # PyTorch SDPA: может уходить в FlashAttention / memory-efficient kernel
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)
Здесь два практических решения: RMSNorm как отдельный модуль и F.scaled_dot_product_attention — единый интерфейс, который в зависимости от GPU и библиотек уходит в FlashAttention-2/3 или memory-efficient kernel PyTorch.

1.6 Резюме
Современный LLM — тот же transformer, но с RMSNorm, RoPE, SwiGLU и GQA.

Pretraining учит языку и миру, post-training — следованию инструкциям и предпочтениям.

Код LLM на PyTorch укладывается в сотни строк; сложность — в масштабе данных и инфраструктуре.

Глава 2. Токенизация и подготовка данных
2.1 Зачем нужна токенизация
Модель работает с числами. Токенизатор превращает текст в последовательность целых чисел (token_ids), а обратный декодер — обратно в текст. Качество токенизации напрямую влияет на длину контекста, размер словаря и стоимость обучения.

Простейший вариант — word-level токенизация: каждое слово получает свой ID. Проблемы: огромный словарь, OOV (out-of-vocabulary) слова, неэффективное использование контекста. Современные LLM используют subword-токенизацию: BPE, WordPiece или Unigram. В открытых моделях 2024–2026 чаще всего встречаются BPE (GPT-2/tiktoken) и SentencePiece.

2.2 Byte Pair Encoding (BPE)
BPE начинает с отдельных байтов (или символов) и итеративно сливает самые частые пары в новые токены. Итог — словарь из нескольких десятков тысяч токенов. Любое слово можно представить как последовательность сабвордов, поэтому OOV-проблемы исчезают.

Пример работы BPE (упрощённо):

Начальный словарь: {a, b, c, ..., z, _, ...}.

Часто встречающаяся пара t+h сливается в токен th.

Затем th+e → the, и так далее.

В современных реализациях (tiktoken, HuggingFace tokenizers) BPE работает на байтовом уровне и поддерживает регулярные выражения для предварительного разбиения текста.

2.3 Использование tiktoken
python
# pip install tiktoken
import tiktoken

enc = tiktoken.get_encoding("gpt2")
ids = enc.encode("Привет, мир! Hello, world.")
print(ids)
print(enc.decode(ids))
Для Llama/Qwen обычно используется SentencePiece или HuggingFace-токенизатор из AutoTokenizer. Для GPT-2/GPT-4 — tiktoken.

2.4 Специальные токены
<|endoftext|> — разделитель документов (GPT-2).

<s>, </s> — BOS/EOS (Llama, Mistral).

<|user|>, <|assistant|>, <|im_end|> — роли в chat-моделях (Qwen, ChatML).

<pad> — паддинг для батчей переменной длины.

Важно: padding-токены маскируются в loss и attention, иначе модель учится предсказывать мусор.

2.5 Датасет и DataLoader
Ниже — рабочий пример PyTorch-датасета, который режет корпус на блоки фиксированной длины (для pretraining).

python
import torch
from torch.utils.data import Dataset, DataLoader

class CausalLMDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, block_size: int, stride: int | None = None):
        self.tokens = token_ids
        self.block_size = block_size
        self.stride = stride or block_size

    def __len__(self):
        return max(0, (len(self.tokens) - self.block_size - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.tokens[start : start + self.block_size]
        y = self.tokens[start + 1 : start + self.block_size + 1]
        return x, y

def build_dataloader(text: str, enc, block_size=1024, batch_size=8, stride=None):
    ids = torch.tensor(enc.encode(text), dtype=torch.long)
    ds = CausalLMDataset(ids, block_size=block_size, stride=stride)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
Ключевые моменты:

y сдвинут на 1 относительно x — это next-token prediction.

stride < block_size даёт перекрывающиеся окна (больше данных, больше переобучения).

drop_last=True убирает неполные батчи.

2.6 Паддинг и attention-маска для SFT
Для SFT батчи содержат последовательности разной длины. Паддинг добавляет <pad> справа, а attention-маска запрещает attend на padding.

python
def collate_sft(batch, pad_id: int):
    max_len = max(len(x) for x, _ in batch)
    xs, ys, masks = [], [], []
    for x, y in batch:
        pad = max_len - len(x)
        xs.append(x + [pad_id] * pad)
        ys.append(y + [-100] * pad)   # -100 игнорируется в cross_entropy
        masks.append([1] * len(x) + [0] * pad)
    return (
        torch.tensor(xs),
        torch.tensor(ys),
        torch.tensor(masks, dtype=torch.bool),
    )
В cross_entropy PyTorch игнорирует значения -100, поэтому loss считается только по реальным токенам.

2.7 Упражнения
Сравните размеры словаря tiktoken gpt2 и токенизатора Llama 3. Почему у них разные размеры?

Реализуйте CausalLMDataset со stride=block_size//2. Как меняется число примеров?

Для chat-формата составьте шаблон ChatML и проверьте, что роли кодируются отдельными токенами.

2.8 Резюме
BPE/SentencePiece — стандарт токенизации LLM.

Специальные токены управляют ролями, границами и padding.

Next-token dataset — сдвиг на 1; padding-маска обязательна для SFT.

Глава 3. Внимание: от dot-product до FlashAttention
3.1 Зачем нужно внимание
В предложении «Кот сел на ковёр, потому что он был тёплым» модель должна понять, что «он» относится к «ковру», а не к «коту». Attention позволяет каждому токену «смотреть» на другие токены и собирать контекст.

3.2 Self-attention
Для входа 
X
∈
R
T
×
d
X∈R 
T×d
  вычисляем:

Q
=
X
W
Q
,
K
=
X
W
K
,
V
=
X
W
V
Q=XW 
Q
​
 ,K=XW 
K
​
 ,V=XW 
V
​
 
A
t
t
e
n
t
i
o
n
(
Q
,
K
,
V
)
=
s
o
f
t
m
a
x
 ⁣
(
Q
K
⊤
d
k
+
M
)
V
Attention(Q,K,V)=softmax( 
d 
k
​
 
​
 
QK 
⊤
 
​
 +M)V
где 
M
M — causal-маска: 
M
i
j
=
−
∞
M 
ij
​
 =−∞ для 
j
>
i
j>i и 
0
0 иначе.

Масштабирование 
d
k
d 
k
​
 
​
  стабилизирует softmax: без него при больших 
d
k
d 
k
​
  softmax вырождается в «почти one-hot», градиенты становятся крошечными.

3.3 Наивная реализация
python
def attention_naive(q, k, v, causal=True):
    # q,k,v: (B, H, T, d)
    scale = q.size(-1) ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        T = q.size(-2)
        mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v
Эта версия понятна, но неэффективна: матрица 
T
×
T
T×T для каждой головы; при 
T
=
32
k
T=32k и 
H
=
32
H=32 это гигабайты.

3.4 PyTorch SDPA
С версии 2.0 PyTorch предоставляет F.scaled_dot_product_attention, который автоматически выбирает FlashAttention, memory-efficient kernel или naive-реализацию в зависимости от железа и параметров.

python
import torch.nn.functional as F

def attention_sdpa(q, k, v):
    # q,k,v: (B, H, T, d); GQA-повтор K/V делается до вызова
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
Плюсы:

Нет материализации 
T
×
T
T×T матрицы в HBM (FlashAttention держит тайлы в SRAM).

Значительное ускорение на длинных контекстах.

3.5 Как работает FlashAttention (концептуально)
FlashAttention никогда не записывает полную матрицу attention в память. Он блочно (tile-by-tile) проходит по 
Q
Q, 
K
K, 
V
V, вычисляет локальные softmax и накапливает результат:

Разбиваем 
Q
Q, 
K
K, 
V
V на блоки, помещающиеся в SRAM.

Для каждого блока 
Q
Q идём по блокам 
K
K, 
V
V; вычисляем 
S
=
Q
K
⊤
S=QK 
⊤
 , локальный softmax-статистики (max, sum), накапливаем.

Используем online softmax: обновляем running max и running sum по мере прохода.

Записываем только итоговый 
O
O, а не промежуточные матрицы.

Итог: 
O
(
T
)
O(T) памяти вместо 
O
(
T
2
)
O(T 
2
 ), на порядок быстрее для длинных контекстов. FlashAttention-2 дополнительно оптимизирует распараллеливание и снижает объём пересчётов; FlashAttention-3 учитывает архитектуру Hopper (H100).

3.6 Causal-маска и GQA в SDPA
При GQA перед вызовом SDPA повторяем K и V, чтобы число голов совпало с Q:

python
def gqa_attention(q, k, v, n_heads, n_kv_heads, is_causal=True):
    # q: (B, n_heads, T, d), k/v: (B, n_kv_heads, T, d)
    repeat = n_heads // n_kv_heads
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
На практике в проде часто пишут вручную объединённый QKV-проекцией и делают reshape — это экономит память и ускоряет инференс.

3.7 Упражнения
Реализуйте наивное attention и сравните результат с SDPA.

Замерьте время и память при 
T
=
1024
T=1024 и 
T
=
8192
T=8192 на CPU и на GPU.

Объясните, почему flash-attn несовместим с произвольными масками без модификации.

3.8 Резюме
Self-attention — основа transformer.

Масштабирование и causal-маска — стандарт для LLM.

SDPA и FlashAttention — практический путь к длинным контекстам.

Глава 4. Современный transformer block: RMSNorm, RoPE, SwiGLU, GQA
4.1 Общая схема
Современный блок (pre-norm):

text
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
Оба residual-соединения обязательны: они смягчают vanishing gradients и позволяют строить очень глубокие сети.

4.2 RMSNorm
Уже реализован в главе 1. Повторим формулу:

R
M
S
N
o
r
m
(
x
)
=
x
1
d
∑
x
i
2
+
ϵ
⊙
γ
RMSNorm(x)= 
d
1
​
 ∑x 
i
2
​
 +ϵ
​
 
x
​
 ⊙γ
В современных моделях 
γ
γ почти всегда инициализируется единицами. Никакого 
β
β нет.

4.3 RoPE
Позиция кодируется вращением пар компонент. Для 
d
d-мерного вектора разбиваем его на 
d
/
2
d/2 пар 
(
x
2
i
,
x
2
i
+
1
)
(x 
2i
​
 ,x 
2i+1
​
 ) и вращаем на угол 
θ
i
⋅
m
θ 
i
​
 ⋅m:

(
x
2
i
′
x
2
i
+
1
′
)
=
(
cos
⁡
(
m
θ
i
)
−
sin
⁡
(
m
θ
i
)
sin
⁡
(
m
θ
i
)
cos
⁡
(
m
θ
i
)
)
(
x
2
i
x
2
i
+
1
)
( 
x 
2i
′
​
 
x 
2i+1
′
​
 
​
 )=( 
cos(mθ 
i
​
 )
sin(mθ 
i
​
 )
​
  
−sin(mθ 
i
​
 )
cos(mθ 
i
​
 )
​
 )( 
x 
2i
​
 
x 
2i+1
​
 
​
 )
где 
θ
i
=
10000
−
2
i
/
d
θ 
i
​
 =10000 
−2i/d
 . В Llama 3 используется база 500000 (для длинного контекста), а YaRN модифицирует частоты для экстраполяции.

Реализация:

python
def precompute_rope(dim, max_seq_len, base=10000.0, device="cpu"):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(t, freqs)  # (T, dim/2)
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin):
    # x: (B, H, T, d)
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos = cos[: x.shape[-2]].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.shape[-2]].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated
Важно: RoPE применяется и к Q, и к K, но не к V.

4.4 SwiGLU
python
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_mult=4):
        super().__init__()
        hidden = int(hidden_mult * dim)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
F.silu — это 
S
w
i
s
h
(
x
)
=
x
σ
(
x
)
Swish(x)=xσ(x). На практике hidden_mult берут ≈ 8/3, чтобы число параметров совпало с GELU-FFN на 4×.

4.5 Полный transformer block
python
class ModernBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.emb_dim)
        self.attn = CausalSelfAttention(cfg)  # из главы 1
        self.norm2 = RMSNorm(cfg.emb_dim)
        self.ffn = SwiGLU(cfg.emb_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
4.6 Упражнения
Замените RoPE на обучаемые позиционные эмбеддинги и сравните качество на маленьком датасете.

Реализуйте precompute_rope с базой 500000 (как в Llama 3).

Попробуйте hidden_mult=8/3 и сравните число параметров с GELU-FFN.

4.7 Резюме
Pre-norm + residual — стандарт глубины.

RoPE даёт относительные позиции, совместимые с FlashAttention.

SwiGLU и GQA — дефолт для современных LLM.

Глава 5. Сборка LLM и pretraining
5.1 Полная модель
python
class MiniLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.blocks = nn.ModuleList([ModernBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.emb_dim)
        self.lm_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            return logits, loss
        return logits, None
5.2 Weight tying
Эмбеддинги токенов и LM-head делят одну матрицу: self.lm_head.weight = self.tok_emb.weight. Это экономит ~30% параметров для маленьких моделей и часто улучшает генерализацию.

python
self.lm_head.weight = self.tok_emb.weight  # weight tying
5.3 Init и стабильность
Веса линейных слоёв — N(0, 0.02) (как в GPT-2).

Residual-пути масштабируют init на 
1
/
2
L
1/ 
2L
​
 , чтобы сумма дисперсий не росла с глубиной.

Эмбеддинги — маленькие случайные числа.

python
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
5.4 Training loop
python
import math, time

def train(model, loader, optimizer, device, epochs, grad_clip=1.0, log_every=50):
    model.train()
    step = 0
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if step % log_every == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            step += 1
5.5 Learning rate schedule
Современный стандарт: linear warmup + cosine decay.

python
def lr_at(step, base_lr, warmup, total):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * progress))
Cosine decay плавно уменьшает шаг к концу обучения — это снижает риск перелёта через минимум.

5.6 Оценка: loss и perplexity
p
e
r
p
l
e
x
i
t
y
=
e
l
o
s
s
perplexity=e 
loss
 
Perplexity интерпретируется как «эффективный размер словаря, по которому модель не уверена». Полезно сравнивать checkpoints между собой.

5.7 Генерация: temperature, top-k, top-p
python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.cfg.context_length:])
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = sorted_logits.softmax(-1)
            cum = probs.cumsum(-1)
            mask = cum - probs > top_p
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
        probs = logits.softmax(-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_tok], dim=1)
    return idx
temperature < 1 — более детерминированно, > 1 — более креативно.

top_k ограничивает выбор k самыми вероятными токенами.

top_p (nucleus) — минимальный набор токенов, чья суммарная вероятность ≥ p.

5.8 Загрузка весов GPT-2 / Llama
Для GPT-2 веса доступны через transformers:

python
from transformers import AutoModelForCausalLM
hf = AutoModelForCausalLM.from_pretrained("gpt2")
state = hf.state_dict()
# вручную переложить в нашу модель с учётом разных имён
Для Llama/Qwen аналогично; главное — совпадают ли n_kv_heads, rope_base, hidden_mult и порядок слоёв. Всегда сверяйте имена параметров и их формы.

5.9 Упражнения
Добавьте weight tying и сравните число параметров.

Попробуйте warmup=200, cosine-schedule и сравните loss с постоянным LR.

Сгенерируйте текст с top_p=0.9, temperature=0.8.

5.10 Резюме
Pretraining = next-token cross-entropy на больших данных.

Weight tying, правильная инициализация и cosine-schedule — стандарт.

Generation-стратегии управляют разнообразием и когерентностью.

Глава 6. Эффективное обучение: mixed precision, FSDP, torch.compile
6.1 Mixed precision (bf16)
Большинство современных LLM обучают в bf16: числа занимают вдвое меньше памяти, а динамический диапазон совпадает с fp32 (в отличие от fp16). В PyTorch достаточно torch.autocast:

python
from torch.amp import autocast, GradScaler

scaler = GradScaler(enabled=False)  # для bf16 scaler не нужен

with autocast(device_type="cuda", dtype=torch.bfloat16):
    logits, loss = model(x, y)
loss.backward()
optimizer.step()
bf16 не требует scaler, fp16 — требует.

6.2 Gradient checkpointing
Позволяет торговать вычисления на память: сохраняем не все активации, а пересчитываем их на backward.

python
from torch.utils.checkpoint import checkpoint

class ModernBlock(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
Память падает в 2–3 раза, шаг становится на 20–30% медленнее — часто выгодно.

6.3 torch.compile
python
model = torch.compile(model, mode="max-autotune")
На современных GPU это даёт 10–40% ускорения за счёт fusion-оптимизаций; на CPU — меньше. Первый шаг компиляции долгий, поэтому используйте в долгих прогонах.

6.4 Distributed Data Parallel (DDP)
DDP реплицирует модель на каждый GPU, деля батч на части. Градиенты усредняются через all-reduce.

python
# torchrun --nproc_per_node=4 train.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
model = model.to(rank)
model = DDP(model, device_ids=[rank])
6.5 FSDP / ZeRO
Когда модель не влезает в память одного GPU, DDP не помогает. FSDP (Fully Sharded Data Parallel) шардирует параметры, градиенты и состояния оптимизатора по GPU. Это аналог ZeRO-3 DeepSpeed.

python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(model, use_orig_params=True)
Комбинация bf16 + FSDP + gradient checkpointing — стандарт обучения 7B–70B моделей на кластере 8×A100/H100.

6.6 Мониторинг
Loss и perplexity — базовая линия.

Grad norm — должна быть стабильной; всплески сигнализируют о проблемах.

Learning rate — визуализируйте schedule.

Скорость (tokens/sec) — узкое место чаще всего data loader, а не GPU.

6.7 Упражнения
Включите bf16 и сравните скорость/память.

Запустите gradient checkpointing на модели с n_layers=24 и сравните память.

Соберите DDP-скрипт для двух GPU.

6.8 Резюме
bf16 — почти бесплатное ускорение.

Gradient checkpointing — способ влезть в память.

FSDP/ZeRO — для больших моделей.

Глава 7. SFT, LoRA/QLoRA и выравнивание: DPO/GRPO
7.1 SFT (Supervised Fine-Tuning)
SFT учит модель следовать инструкциям на парах «запрос → ответ». Формат — chat-шаблон, например ChatML:

text
<|im_start|>user
Как дела?<|im_end|>
<|im_start|>assistant
Отлично, спасибо!<|im_end|>
Loss считается только по токенам ответа ассистента (или с меньшим весом по user-токенам). Это резко повышает качество инструкций.

python
def sft_loss(logits, labels, assistant_mask):
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none",
    )
    mask = assistant_mask.view(-1).float()
    return (loss * mask).sum() / mask.sum().clamp_min(1)
7.2 Полный fine-tuning vs параметр-эффективный
Полный fine-tuning 7B-модели требует ~120 GB памяти на оптимизатор и активации. LoRA резко это снижает.

LoRA
Замороженные веса 
W
W + низкоранговое обновление:

W
′
=
W
+
α
r
B
A
W 
′
 =W+ 
r
α
​
 BA
где 
A
∈
R
r
×
d
A∈R 
r×d
 , 
B
∈
R
d
×
r
B∈R 
d×r
 , 
r
≪
d
r≪d. Обычно 
r
=
8
r=8–
32
32, 
α
=
16
α=16–
32
32.

python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        d_in, d_out = base.in_features, base.out_features
        self.A = nn.Parameter(torch.zeros(r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, r))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)
LoRA обычно ставят на q_proj, k_proj, v_proj, o_proj, а иногда и на gate/up/down FFN.

QLoRA
QLoRA = LoRA + 4-bit квантизация базовой модели (NF4) + double quantization. Позволяет дообучать 7B на одной 24 GB GPU.

python
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
7.3 Выравнивание: DPO и GRPO
DPO (Direct Preference Optimization)
DPO обучает модель на парах «предпочтённый ответ 
y
w
y 
w
​
  / отвергнутый 
y
l
y 
l
​
 ». Вместо RL используется прямой лосс:

L
D
P
O
=
−
log
⁡
σ
 ⁣
(
β
[
log
⁡
π
θ
(
y
w
∣
x
)
π
ref
(
y
w
∣
x
)
−
log
⁡
π
θ
(
y
l
∣
x
)
π
ref
(
y
l
∣
x
)
]
)
L 
DPO
​
 =−logσ(β[log 
π 
ref
​
 (y 
w
​
 ∣x)
π 
θ
​
 (y 
w
​
 ∣x)
​
 −log 
π 
ref
​
 (y 
l
​
 ∣x)
π 
θ
​
 (y 
l
​
 ∣x)
​
 ])
ref — это SFT-модель, замороженная. Ключевые параметры: 
β
∈
[
0.01
,
0.5
]
β∈[0.01,0.5], небольшой LR.

GRPO (Group Relative Policy Optimization)
GRPO — вариант PPO без отдельного value-критика: награда нормируется по группе ответов на один промпт. Стало популярным для reasoning-моделей (DeepSeek-R1 и др.).

Идея:

Для каждого промпта генерируем 
G
G ответов.

Считаем награду 
r
i
r 
i
​
  (reward model или верификатор).

Нормируем: 
r
^
i
=
(
r
i
−
m
e
a
n
)
/
s
t
d
r
^
  
i
​
 =(r 
i
​
 −mean)/std.

Обновляем политику с PPO-клиппингом, используя 
r
^
i
r
^
  
i
​
  как advantage.

Это дешевле PPO (не нужен value-критик) и хорошо масштабируется на маленькие reward-сигналы.

7.4 Практический чек-лист SFT/LoRA
Обучайте на токенах ассистента, а не на всей последовательности.

LR для LoRA — 1e-4…5e-4; для полного fine-tuning — 1e-5…3e-5.

Warmup 1–3% шагов, cosine decay.

Не забывайте pad_token_id и assistant_mask.

Валидация — отдельный held-out набор инструкций, не пересекающийся с train.

7.5 Упражнения
Соберите SFT-датасет из 100 пар в ChatML и обучите LoRA-модель.

Замерьте GPU-память при LoRA 
r
=
8
r=8 и 
r
=
64
r=64.

Реализуйте DPO-loss и проверьте, что при 
π
=
π
ref
π=π 
ref
​
  лосс равен 
log
⁡
2
log2.

7.6 Резюме
SFT — первый и обязательный этап post-training.

LoRA/QLoRA — дешёвая адаптация под домен.

DPO/GRPO — выравнивание под предпочтения и reasoning.

Глава 8. Инференс: KV-cache, квантизация, speculative decoding
8.1 KV-cache
При генерации каждый новый токен требует attention ко всем предыдущим. Наивно — O(T²) вычислений. KV-cache сохраняет K и V предыдущих токенов, поэтому на каждом шаге нужно только обновить Q для нового токена и добавить новые K/V.

python
# Идея: держим K/V в буфере (B, H, T, d), растущем на 1 токен за шаг
# На шаге t: q_t attends to cached K[:t], V[:t]
Эффект: сложность шага становится O(T), а не O(T²). Но память растёт линейно:

K
V
 
b
y
t
e
s
=
2
⋅
B
⋅
T
⋅
H
k
v
⋅
d
⋅
b
y
t
e
s
_
p
e
r
_
e
l
e
m
e
n
t
KV bytes=2⋅B⋅T⋅H 
kv
​
 ⋅d⋅bytes_per_element
Для Llama 3 8B (
H
k
v
=
8
H 
kv
​
 =8, 
d
=
128
d=128, 
T
=
8192
T=8192, bf16) это ≈ 1 GB на последовательность; с GQA — в 4 раза меньше, чем у MHA. Отсюда популярность GQA и MLA.

8.2 PagedAttention
vLLM предложил PagedAttention: KV-cache хранится блоками, как страницы виртуальной памяти. Это устраняет фрагментацию и позволяет эффективно батчить запросы с разной длиной.

8.3 Квантизация
Три основных сценария:

Post-training quantization (PTQ):

GPTQ, AWQ — 4-bit веса, калибровка на маленьком датасете.

bitsandbytes NF4 (для QLoRA).

Quantization-aware training (QAT):

Модель обучается с учётом будущей квантизации.

FP8 на H100:

torch.float8_e4m3fn для активаций и весов; требует поддержки железа.

Правило большого пальца: 4-bit веса + bf16 активации — хороший компромисс для локального инференса; 8-bit почти без потерь; 4-bit иногда теряет 1–3% на бенчмарках.

8.4 Speculative decoding
Идея: маленькая draft-модель генерирует 
K
K токенов, большая модель проверяет их за один forward. Принятые токены остаются, отвергнутые пересчитываются. Ускорение 1.5–3× при сохранении распределения большой модели.

python
# Псевдокод
draft_tokens = draft.generate(prompt, k=5)
verify_logits = target.forward(prompt + draft_tokens)
accepted = sample_accept(draft_tokens, verify_logits)
prompt = prompt + accepted + resample_one(verify_logits)
Требует, чтобы draft и target использовали одинаковый токенизатор.

8.5 Батчинг и continuous batching
vLLM/TGI/TensorRT-LLM используют continuous batching: запросы присоединяются к батчу и покидают его по мере завершения, а не ждут медленных. Это в разы повышает throughput по сравнению с наивным батчингом.

8.6 Практический стек инференса
llama.cpp — CPU/Metal/GPU, квантизация GGUF.

vLLM — production GPU-инференс, PagedAttention, continuous batching.

TensorRT-LLM — максимальная скорость на NVIDIA.

MLX — Apple Silicon.

8.7 Упражнения
Реализуйте KV-cache в своей MiniLLM и замерьте ускорение на 
T
=
512
T=512.

Сравните качество GPTQ-4bit и bf16 на маленьком бенчмарке.

Реализуйте простой speculative decoding с draft-моделью в 4 раза меньше.

8.8 Резюме
KV-cache — фундамент быстрой генерации.

GQA/MLA/квантизация/PagedAttention — пути к длинному контексту и экономии памяти.

Speculative decoding — ускорение без потери качества.

Приложение A. PyTorch для LLM
A.1 Тензоры
python
import torch
x = torch.tensor([[1., 2.], [3., 4.]])
print(x.shape, x.dtype, x.device)
Ключевые операции: view, reshape, transpose, permute, @, einsum, softmax, log_softmax.

A.2 Autograd
python
w = torch.tensor([2.0], requires_grad=True)
loss = (w * 3 - 6).pow(2)
loss.backward()
print(w.grad)  # d loss / d w
torch.no_grad() отключает построение графа — обязательно для инференса.

A.3 nn.Module
python
class MyBlock(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)
    def forward(self, x):
        return self.lin(x)
A.4 DataLoader
python
from torch.utils.data import Dataset, DataLoader
Стандартный паттерн: __init__, __len__, __getitem__.

A.5 Training loop
См. главу 5. Ключевые шаги: zero_grad, forward, loss.backward, step.

A.6 GPU
python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x = x.to(device)
Для Apple Silicon: torch.device("mps").

Приложение B. Разбор архитектур: Llama 3, Qwen, DeepSeek-V3, Gemma
B.1 Llama 3
8B / 70B, context 128k.

Pre-norm RMSNorm, RoPE base 500000, SwiGLU.

GQA: 32 Q-голов, 8 KV-голов.

Токенизатор — tiktoken-style BPE, словарь ~128k.

Fine-tuning: SFT + DPO + safety-фильтры.

B.2 Qwen 2.5/3
Серия от 0.5B до 235B (MoE для флагмана).

GQA, RoPE с YaRN (для 128k–1M контекста).

ChatML-шаблон, сильная мультиязычность.

Qwen3 добавляет thinking-режим (reasoning).

B.3 DeepSeek-V3
MoE 671B (37B активных).

Multi-head Latent Attention (MLA) — низкоранговое сжатие KV.

FP8-обучение и inference.

GRPO для reasoning (DeepSeek-R1).

B.4 Gemma 2/3
RMSNorm, GeGLU (в Gemma 2 — GeGLU, не SwiGLU), GQA, sliding-window attention (чередование глобального и локального).

Дистилляция из больших моделей.

Приложение C. Упражнения
Каждая глава содержит задания. Решения можно оформлять в отдельном блокноте; ключевые подсказки:

Глава 2. tiktoken gpt2 — 50257, Llama 3 — ~128k; разница из-за лучшей компрессии и мультиязычности.

Глава 3. FlashAttention требует особого layout; произвольные маски ломают блочность.

Глава 4. hidden_mult=8/3 компенсирует 3 матрицы SwiGLU против 2 у GELU-FFN.

Глава 5. Weight tying снижает параметры на ~30% для маленьких моделей.

Глава 6. Gradient checkpointing экономит память за счёт ~20% времени.

Глава 7. DPO-лосс при 
π
=
π
ref
π=π 
ref
​
  равен 
log
⁡
2
log2 — полезная проверка.

Глава 8. KV-cache ускоряет шаг с O(T²) до O(T) при линейной памяти.

Конец книги. Материал можно свободно использовать и адаптировать. Если нужно, можно дальше расширить любую главу (например, добавить раздел про MoE или полный проект SFT + DPO на открытых данных).