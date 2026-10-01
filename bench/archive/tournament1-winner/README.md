# Турнир 1: Grand-Champion (Кандидат 12)

**Дата:** 30.09.2026
**Движок:** Strata 0.1.22
**Модель:** Qwen 3.8 Flash Next Coder IQ1_M (125B MoE)
**Кандидатов:** 12 (нумерация 0–11 + Grand-Champion синтез)

## Метрики Победителя
| Метрика | Значение |
|:---|:---:|
| Decode | **5.8 tok/s** |
| Prefill | 7.8 tok/s (>450 из кэша) |
| VRAM Hit Rate | 6.2–8.6% |
| MTP Acceptance | **89.8%** |
| Expert Slots | 275 |
| Score (формула T1) | **23.04** |

## Ключевые параметры (отличия от Baseline)
```
--expert-cache 220          ← было 180
--vram-reserve-mib 480      ← было ~700 (дефолт)
--spec 3                    ← было 4
--spec-min-p 0.82           ← было 0.75
--suffix-draft 3            ← НОВЫЙ
--prompt-cache 16           ← НОВЫЙ
--prompt-cache-root 16      ← НОВЫЙ
--prompt-cache-every 128    ← НОВЫЙ
```

## Ключевые открытия
1. **WDDM PCIe Bottleneck** — `--short-read 2048` критически важен на 4GB ноутбуке
2. **Prompt Cache** — многоуровневый RAM-кэш ускорил повторные чтения в 60+ раз
3. **VRAM Reserve снижение** — 700→480 МБ → +68% слотов экспертов (163→275)
4. **Spec 3 + min-p 0.82** — оптимальный баланс acceptance/overhead (+32% decode)

## Файлы
- [strata-coder-iq1_m.tournament1-champion.json](file:///c:/Users/Ilia%20V/Documents/antigravity/calm-noether/Strata/bench/archive/tournament1-winner/strata-coder-iq1_m.tournament1-champion.json)
- [Полная матрица Турнира 1](file:///c:/Users/Ilia%20V/Documents/antigravity/calm-noether/Strata/bench/archive/tournament1-winner/tournament1_matrix.md)
