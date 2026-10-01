# Конфигурация до турниров (Baseline / Pre-Tournament)

**Дата:** до 30.09.2026
**Движок:** Strata 0.1.22
**Модель:** Qwen 3.8 Flash Next Coder IQ1_M (125B MoE)

## Метрики Baseline
| Метрика | Значение |
|:---|:---:|
| Decode | 4.3–4.5 tok/s |
| Prefill | 7.3 tok/s |
| VRAM Hit Rate | 3.4–4.3% |
| MTP Acceptance | 69.6–75.1% |
| Expert Slots | 163 |
| VRAM Reserve | 700 MiB (дефолт) |

## Ключевые параметры
```
--expert-cache 180
--expert-cache-cpu-order
--adapt-every 0
--adapt-swaps 0
--prefill 1024
--spec 4
--spec-min-p 0.75
--mtp-window 4096
--max-context 65536
--kv q4_0
--pcie-frac 0
--short-read 2048
```

## Проблемы
- Нет prompt-cache → пересчёт контекста при каждом сообщении
- Высокий vram-reserve (700 MiB) → всего 163 слота
- spec 4 + min-p 0.75 → MTP acceptance только 69–75%
- Нет suffix-draft
- Нет KV-streaming

## Файл конфигурации
[strata-coder-iq1_m.pre-tournament.json](file:///c:/Users/Ilia%20V/Documents/antigravity/calm-noether/Strata/bench/archive/pre-tournament/strata-coder-iq1_m.pre-tournament.json)
