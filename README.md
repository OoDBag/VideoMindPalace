# VideoMindPalace
[CVPR 2025] The official implementation of the paper "Building a Mind Palace: Structuring Environment-Grounded Semantic Graphs for Effective Long Video Analysis with LLMs" 

---

## ⚙️ Environment Setup

```bash
# Create and activate the environment
conda create -n mindpalace python=3.9
conda activate mindpalace

# Install dependencies
pip install openai
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas
pip install transformers==4.28.1
pip install accelerate
```

---

## 🧭 Full Pipeline Overview

### 📦 1. Preprocessing and Tracking Extraction (EgoSchema)

We use [AMEGO](https://github.com/gabrielegoletto/AMEGO)'s tracking pipeline to extract per-frame object trajectories from EgoSchema videos.

```bash
# Follow AMEGO's official instructions to obtain tracking outputs
```

---

### 🧱 2. Tracking Object Classification and clustering

```bash
python cluster_class.py
python cluster.py
```

---

### 📝 3. Caption Generation

```bash
python caption.py
```

---

### 🕸️ 4. Graph Construction

```bash
python build_graph.py
```

---

### ❓ 5. Graph-based Question Answering

```bash
sh egoschema_qa.sh
```

---

## 📜 Citation

If you find this work useful, please consider citing:

```bibtex
@article{huang2025building,
  title={Building a Mind Palace: Structuring Environment-Grounded Semantic Graphs for Effective Long Video Analysis with LLMs},
  author={Huang, Zeyi and Ji, Yuyang and Wang, Xiaofang and Mehta, Nikhil and Xiao, Tong and Lee, Donghyun and Vanvalkenburgh, Sigmund and Zha, Shengxin and Lai, Bolin and Yu, Licheng and others},
  journal={arXiv preprint arXiv:2501.04336},
  year={2025}
}```
