# Итоговая Таблица Лидеров: Олимпийский Турнир 3.0

**Дата завершения:** 2026-09-30 22:16:50  
**Модель:** `Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M` (125B MoE, 256/512 экспертов)  
**Железо:** NVIDIA GeForce RTX 3050 Laptop GPU (4 ГБ GDDR6) + AMD Ryzen 5 5500U (6C/12T) + 32 ГБ DDR4-3200 + NVMe SSD  
**Контекст:** `--max-context 65536` (Правило §0 соблюдено на 100%)

| Ранг | ID | Название конфигурации | Категория | Статус | Decode (tok/s) | Prefill (tok/s) | Hit Rate (%) | MTP Acc (%) | Слоты VRAM | Quality Бонус | Итоговый Score |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **1** | `OGC3` | **Olympic-Grand-Champion-3.0** | **Grand Synthesis of 4 Olympians** | **PASSED (Full 3/3)** | **4.9** | **7.9** | **22.6%** | **85.7%** | **385** | **+10.0** | **96.51** |
| 🥈 **2** | `O03` | Olympian-Gamma-Speculation | Speculative Parallelism & N-Gram Gating | PARTIAL (2/3) | 5.0 | 7.9 | 21.3% | 82.7% | 385 | +6.0 | **89.95** |
| 🥉 **3** | `O01` | Olympian-Alpha-Frontier | Frontier Network Intelligence & Fast DMA Adaptation | PARTIAL (1/3) | 4.8 | 7.9 | 22.6% | 84.5% | 385 | +3.0 | **88.83** |
| 4 | `O04` | Olympian-Delta-Holistic | Cross-Engine Transfer & Holistic Synthesis | PARTIAL (2/3) | 4.9 | 7.9 | 20.7% | 82.5% | 385 | +6.0 | **88.22** |
| 5 | `O02` | Olympian-Beta-Memory | Memory Hierarchy & Fast Swaps Architecture | PARTIAL (1/3) | 5.1 | 7.9 | 20.0% | 80.3% | 385 | +3.0 | **84.50** |

---

### Сравнительный прогресс всех турниров
| Этап эволюции | Decode tok/s | Prefill tok/s | Hit Rate VRAM | MTP Acceptance | Слоты VRAM | Итоговый Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (До турниров)** | 4.3 | 6.5 | 3.4% | ~70% | 163 | 12.50 |
| **Турнир 1 (Grand-Champion)** | 5.8 (короткий) | 6.8 | 7.4% | 89.8% | 275 | 23.04 (формула T1) |
| **Турнир 2 (Grand-Champion GC2b)** | 5.6 | 7.0 | 13.0% | 87.7% | 375 | 80.66 |
| **Олимпийский Турнир 3.0 (OGC3)** | **4.9 (complex 3-tier)** | **7.9** | **22.6%** | **85.7%** | **385** | **96.51** |