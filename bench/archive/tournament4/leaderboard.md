# Итоговая Таблица Лидеров: Турнир 4.0

**Дата завершения:** 2026-09-30 23:05:47  
**Модель:** `Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M` (125B MoE, 256/512 экспертов)  
**Железо:** RTX 3050 Laptop 4GB + Ryzen 5 5500U + 32GB DDR4-3200 + NVMe SSD  
**Контекст:** `--max-context 65536` (Правило §0 строго соблюдено всеми кандидатами)

| Ранг | ID | Название | Категория | Статус | Decode (tok/s) | Prefill (tok/s) | Hit Rate (%) | MTP Acc (%) | Слоты VRAM | Test B (LRU) | Score |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **1** | `T4_03` | **Speculative-Dual-Tier-Architect** | **Decoupled Speculation (MTP 3 + Suffix 6)** | **PASSED (Full 3/3)** | **5.0** | **9.3** | **24.5%** 🥇 | **84.4%** | **426** 🥇 | **4.86** | **109.84** |
| 🥈 **2** | `T4_01` | **PLE-IO-Architect** | **Zero-Jitter Direct IOCP & VRAM Expansion** | **PASSED (Full 3/3)** | **5.1** | **9.3** | **22.8%** | **82.7%** | **426** | **4.95** 🥇 | **108.12** |
| 🥉 **3** | `T4_02` | **PCIe-DMA-Auditor** | **Balanced MTP Threshold & Empirical Audit** | **PASSED (Full 3/3)** | **5.2** 🥇 | **9.4** 🥇 | **21.4%** | **86.5%** 🥇 | **426** | **4.78** | **107.45** |
| 4 | `T4_04` | Kernel-Topology-Architect | Low-Level Windows 11 & Zen 2 Optimization | PASSED (Full 3/3) | 5.0 | 9.3 | 22.3% | 79.8% | 426 | 4.86 | **106.08** |
| 5 | `T4_GC` | Tournament-4-Grand-Champion | Grand Synthesis of All 4 Breakthroughs | PASSED (Full 3/3) | 5.0 | 8.4 | 22.3% | 81.0% | 426 | 4.94 | **105.75** |

---

### Сравнительный прогресс всех 4 этапов эволюции

| Параметр | Baseline (До турниров) | Турнир 1 (Grand-Champion) | Турнир 2 (GC2b) | Олимпийский 3.0 (OGC3) | **Турнир 4.0 (T4_03 / T4_01)** | Прирост от Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Слотов экспертов в VRAM** | 163 слота | 275 слотов | 375 слотов | 385 слотов | **426 слотов** 🏆 | **+161% (+263 слота)** |
| **VRAM Expert Hit Rate** | 3.4% | 7.4% | 13.0% | 22.6% | **24.5%** 🏆 | **В 7.2 раза выше!** |
| **Prefill Speed** | 6.5 tok/s | 6.8 tok/s | 7.0 tok/s | 7.9 tok/s | **9.4 tok/s** 🏆 | **+44.6%** |
| **Decode Speed (Raw Engine)** | 4.3 tok/s | 5.8 tok/s | 5.6 tok/s | 5.0 tok/s | **5.2 tok/s** | **Стабильно при 3-Tier** |
| **LRU Cache Code Generation (Test B)** | ~2.5 tok/s | ~3.8 tok/s | ~4.2 tok/s | 4.58 tok/s | **4.95 tok/s** 🏆 | **Почти удвоение!** |
| **Итоговый балл** | 12.50 | 23.04 | 80.66 | 96.51 | **109.84** 🏆 | **+778%** |
| **Контекстное окно (Правило §0)** | 65536 | 65536 | 65536 | 65536 | **65536** | **100% сохранено** |
