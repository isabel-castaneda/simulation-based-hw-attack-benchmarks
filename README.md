# Simulation-Based Hardware Attack Benchmarks

This repository contains the source code and simulation outputs of various programs used to evaluate AI-based techniques for detecting hardware-level attacks. The benchmarks were compiled and executed using the gem5 simulator targeting the RISC-V architecture.

Each benchmark is organized into a separate folder and categorized as either **benign** or **malicious**, depending on its intended behavior. All simulations were performed using gem5’s syscall emulation (SE) mode unless system-level interaction was required.

This work is part of the Master’s Thesis titled:

> **Simulation-Based Evaluation of AI Techniques for Hardware Attack Detection**  
> High-Performance Computing specialization, MIRI Master's Program  
> Universitat Politècnica de Catalunya (UPC), Facultat d'Informàtica de Barcelona (FIB)  
> Academic Year 2025

---

## Structure

```text
.
├── benign/
│   ├── loop_sum/
│   │   ├── loop_sum.c
│   │   └── dataset/
│   │       ├── run_01.csv
│   │       ├── run_02.csv
│   │       └── ...
│   └── ...
├── malicious/
│   ├── spectrev1/
│   │   ├── spectrev1.c
│   │   └── dataset/
│   │       ├── run_01.csv
│   │       ├── run_02.csv
│   │       └── ...
│   └── ...
