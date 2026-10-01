# Архив оптимизации Strata — Мастер-Индекс

**Модель:** Qwen 3.8 Flash Next Coder IQ1_M (125B MoE, 256/512 экспертов)  
**Железо:** NVIDIA GeForce RTX 3050 Laptop 4GB + AMD Ryzen 5 5500U + 32GB DDR4-3200  
**Движок:** Strata 0.1.22  
**Правило §0:** Контекст `--max-context 65536` полностью сохранён на всех этапах.

---

## Эволюция конфигураций и метрик

```
Pre-Tournament (Baseline)
  │  Decode: 4.3 tok/s | Hit Rate: 3.4% | Slots: 163 | Score: 12.50
  │
  ▼
Tournament 1 Grand-Champion
  │  Decode: 5.8 tok/s | Hit Rate: 6-8% | Slots: 275 | MTP: 89.8% | Score: 23.04 (T1)
  │  Прирост: +35% decode, +68% slots
  │
  ▼
Tournament 2 Grand-Champion V2b (GC2b)
  │  Decode: 5.6 tok/s | Hit Rate: 13.0% | Slots: 375 | MTP: 87.7% | Score: 80.66
  │  Прирост: +92% hit rate, +36% slots, +250% score
  │
  ▼
Олимпийский Турнир 3.0 Grand-Champion (OGC3)
  │  Decode: 4.9 tok/s (3-Tier Sustained) | Prefill: 7.9 tok/s | Hit Rate: 22.6% | Slots: 385 | Score: 96.51
  │
  ▼
Турнир 4.0 Grand-Champion (T4_03 / T4_01)  ← ТЕКУЩАЯ АКТИВНАЯ
     Decode: 5.0–5.2 tok/s (Test B LRU: 4.95 tok/s!) | Prefill: 9.4 tok/s (РЕКОРД!)
     Hit Rate: 24.5% (АБСОЛЮТНЫЙ РЕКОРД!) | Slots: 426 (РЕКОРД!) | Score: 109.84
     Прирост Hit Rate от Baseline: +620% (в 7.2 раза выше!)
```

---

## Папки архива

| Папка | Содержимое |
|:---|:---|
| `bench/archive/pre-tournament/` | Заводская конфигурация до всех оптимизаций |
| `bench/archive/tournament1-winner/` | Победитель Турнира 1 + полная матрица 12 кандидатов |
| `bench/archive/tournament2-winner/` | Победитель Турнира 2 + полная матрица 20 кандидатов |
| `bench/archive/tournament3-olympic/` | Победитель Олимпийского Турнира 3.0 (OGC3) |
| `bench/archive/tournament4/` | **Победитель Турнира 4.0 (T4_03 / T4_01)** + журнал Anti-Falsification |
| `bench/tournament4/reports/` | Полные исследовательские отчёты субагентов T4_01 – T4_04 |
| `bench/tournament4/candidates/` | Все конфигурации кандидатов и результаты бенчмарков |

---

## Быстрый откат к любой конфигурации

```powershell
# 1. Откат к Baseline (до всех турниров):
Copy-Item Strata\bench\archive\pre-tournament\strata-coder-iq1_m.pre-tournament.json Strata\strata-coder-iq1_m.json

# 2. Откат к Победителю Турнира 1:
Copy-Item Strata\bench\archive\tournament1-winner\strata-coder-iq1_m.tournament1-champion.json Strata\strata-coder-iq1_m.json

# 3. Откат к Победителю Турнира 2 (GC2b):
Copy-Item Strata\bench\archive\tournament2-winner\strata-coder-iq1_m.tournament2-champion.json Strata\strata-coder-iq1_m.json

# 4. Откат к Победителю Олимпийского Турнира 3.0 (OGC3):
Copy-Item Strata\bench\archive\tournament3-olympic\strata-coder-iq1_m.tournament3-champion.json Strata\strata-coder-iq1_m.json

# 5. Текущий Победитель Турнира 4.0 (уже активен):
Copy-Item Strata\bench\archive\tournament4\strata-coder-iq1_m.tournament4-champion.json Strata\strata-coder-iq1_m.json
```
