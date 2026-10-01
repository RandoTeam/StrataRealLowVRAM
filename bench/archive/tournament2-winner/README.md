# Турнир 2: Grand-Champion V2b

**Дата:** 30.09.2026
**Движок:** Strata 0.1.22
**Модель:** Qwen 3.8 Flash Next Coder IQ1_M (125B MoE)
**Кандидатов:** 20 субагентов (S01–S20) + Grand-Champion синтез

## Метрики Победителя
| Метрика | T1 Champion | **T2 Champion (GC2b)** | Δ |
|:---|:---:|:---:|:---:|
| Decode | 5.8 tok/s | **5.6 tok/s** | −3% |
| Prefill | 7.8 tok/s | **7.0 tok/s** | −10% |
| VRAM Hit Rate | 6–8% | **13.0%** | **+92%** |
| MTP Acceptance | 89.8% | **87.7%** | −2% |
| Expert Slots | 275 | **375** | **+36%** |
| Score (формула T2) | 73.89 | **80.66** | **+9.2%** |

## Ключевые параметры (отличия от T1 Champion)
```
--expert-cache 450          ← было 220
--vram-reserve-mib 280      ← было 480
--kv-resident 20480         ← НОВЫЙ (KV streaming)
--adapt-every 16            ← было 0
--adapt-swaps 4             ← было 0
--pool-workers 5            ← НОВЫЙ (1:1 mapping на 6 ядер)
--mtp-window 2048           ← было 4096
ENV: STRATA_POOL_SPIN_US=50000  ← НОВЫЙ
```

## Синергии из 5 победителей
| Техника | Источник | Эффект |
|:---|:---:|:---|
| DMA-адаптация кэша | S04 | Hit rate 6%→13% |
| Низкий VRAM reserve | S17/S19 | 275→375 слотов |
| KV-streaming в pinned RAM | S02 | 99.4% KV в VRAM, −338 МБ |
| CPU pool 1:1 ядрам | S09 | Минимум SMT-конфликтов |
| Спинлок 50мс | S09 | Минимум джиттера планировщика |

## Файлы
- [strata-coder-iq1_m.tournament2-champion.json](file:///c:/Users/Ilia%20V/Documents/antigravity/calm-noether/Strata/bench/archive/tournament2-winner/strata-coder-iq1_m.tournament2-champion.json)
- [GC2b_candidate.json](file:///c:/Users/Ilia%20V/Documents/antigravity/calm-noether/Strata/bench/archive/tournament2-winner/GC2b_candidate.json)
- [GC2b_candidate_result.json](file:///c:/Users/Ilia%20V/Documents/antigravity/calm-noether/Strata/bench/archive/tournament2-winner/GC2b_candidate_result.json)
- [leaderboard.json](file:///c:/Users/Ilia%20V/Documents/antigravity/calm-noether/Strata/bench/archive/tournament2-winner/leaderboard.json)
- [leaderboard.md](file:///c:/Users/Ilia%20V/Documents/antigravity/calm-noether/Strata/bench/archive/tournament2-winner/leaderboard.md)
