# Do Audio-Visual Large Language Models Really See and Hear?

<p align="center">
  <a href="https://arxiv.org/abs/2604.02605">
    <img src="https://img.shields.io/badge/arXiv-2604.02605-b31b1b" alt="arXiv"/>
  </a>
  <a href="https://avllm-interpretability.github.io">
    <img src="https://img.shields.io/badge/Project-Website-blue" alt="Website"/>
  </a>
</p>

Code for experiments conducted in the paper, with Qwen 2.5 Omni as the representative model.


## Installation
```bash
pip3 install -r requirements.txt
```

## Experiments

### Logit Lens Experiment
```bash
python src/logitlens_experiment.py \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --video_path assets/02321.mp4
```

### Attention Knockout Experiment
```bash
python src/attention_knockout_experiment.py \
  --model_path Qwen/Qwen2.5-Omni-3B \
  --video_path assets/02321.mp4
```

## Citation

```bibtex
@misc{selvakumar2026audiovisuallargelanguagemodels,
      title={Do Audio-Visual Large Language Models Really See and Hear?},
      author={Ramaneswaran Selvakumar and Kaousheik Jayakumar and S Sakshi and Sreyan Ghosh and Ruohan Gao and Dinesh Manocha},
      year={2026},
      eprint={2604.02605},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.02605},
}
```