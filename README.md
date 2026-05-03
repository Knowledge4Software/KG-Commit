# JIT Defect Prediction Framework

A modular, time-aware framework for Just-in-Time (JIT) software defect prediction at the commit level.

## 🚀 Features

- Temporal-aware streaming (Window abstraction)
- Online and batch learning support
- Plug-and-play feature engineering
- Strict leakage-free evaluation
- Experiment reproducibility

## 🧠 Core Idea

We model the repository as a stream of **Windows**, where each window represents a time-ordered batch of commits:

Raw Commits → Windows → Features → Model → Evaluation

## 📁 Project Structure

(explain briefly, not full tree)

## ⚙️ Installation

```bash
pip install -e .