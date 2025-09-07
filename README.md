# A Transformer Based Handwriting Recognition System Jointly Using Online and Offline Features

[![Conference](https://img.shields.io/badge/ACPR-2025%20Oral-blueviolet)](https://acpr2025.com/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.20255-B31B1B.svg)](https://arxiv.org/abs/2506.20255)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

**Ayush Lodh**¹
<a href="https://orcid.org/0009-0001-7506-0900"><img alt="ORCID" src="https://img.shields.io/badge/ORCID-0009--0001--7506--0900-A6CE39?style=flat&logo=orcid&logoColor=white"></a> †,
**Ritabrata Chakraborty**¹ ²
<a href="https://orcid.org/0009-0009-3597-3703"><img alt="ORCID" src="https://img.shields.io/badge/ORCID-0009--0009--3597--3703-A6CE39?style=flat&logo=orcid&logoColor=white"></a> †,
**Palaiahnakote Shivakumara**³
<a href="https://orcid.org/0000-0001-9026-4613"><img alt="ORCID" src="https://img.shields.io/badge/ORCID-0000--0001--9026--4613-A6CE39?style=flat&logo=orcid&logoColor=white"></a>,
**Umapada Pal**¹
<a href="https://orcid.org/0000-0002-5426-2618"><img alt="ORCID" src="https://img.shields.io/badge/ORCID-0000--0002--5426--2618-A6CE39?style=flat&logo=orcid&logoColor=white"></a>,

¹ CVPR Unit, Indian Statistical Institute, Kolkata, India &nbsp;•&nbsp; <a href="mailto:ayushlodh26@gmail.com">ayushlodh26@gmail.com</a> &nbsp;•&nbsp; <a href="mailto:umapada@isical.ac.in">umapada@isical.ac.in</a>  

² Manipal University Jaipur, India &nbsp;•&nbsp; <a href="mailto:ritabrata.229301716@muj.manipal.edu">ritabrata.229301716@muj.manipal.edu</a>  

³ University of Salford, UK &nbsp;•&nbsp; <a href="mailto:s.palaiahnakote@salford.ac.uk">s.palaiahnakote@salford.ac.uk</a>  


† <em>Work done during internship at ISI Kolkata.</em>


---

<!-- <p align="center">
  <img src="" width="900" alt="Architecture Diagram">
</p> -->

## 🔥 Abstract 

We posit that handwriting recognition benefits from complementary cues carried by the rasterized complex glyph and the pen’s trajectory, yet most systems exploit only one modality. We introduce an end-to-end network that performs early fusion of offline images and online stroke data within a shared latent space. A patch encoder converts the grayscale crop into fixed-length visual tokens, while a lightweight transformer embeds the (x, y, pen) sequence. Learnable latent queries attend jointly to both token streams, yielding context-enhanced stroke embeddings that are pooled and decoded under a cross-entropy loss objective. Because integration occurs before any high-level classification, temporal cues reinforce each other during representation learning, producing stronger writer independence. Comprehensive experiments on IAMOn-DB, and VNOn-DB demonstrate that our approach achieves state-of-the art accuracy, exceeding previous bests by up to 1%. Our study also shows adaptation of this pipeline with gesturification on the ISI-Air dataset.

## 📦 Installation

Coming Soon
---
## 🪪 License

This repository is released under the <strong>MIT License</strong> (add a <code>LICENSE</code> file if not present).  
© 2025 The authors. All rights reserved where applicable.
