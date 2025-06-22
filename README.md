### EHIR
[Paper](https://www.sciencedirect.com/science/article/pii/S0167865524001983)

## Data Preparation
Please make sure that all template and target image pairs are placed together in a single directory (e.g., ./data). Each pair should be named in the following format: {name}_temp.jpg and {name}_test.jpg.

## Usage
You can align your images by running the following script:
```
python tools/align.py
```

## Acknowledgments
This work was contributed equally by Shuixin Deng and Lei Deng.

## Citation
```
@article{deng2024ehir,
  title={EHIR: energy-based hierarchical iterative image registration for accurate PCB defect detection},
  author={Deng, Shuixin and Deng, Lei and Meng, Xiangze and Sun, Ting and Chen, Baohua and Chen, Zhixiang and Hu, Hao and Xie, Yusen and Yin, Hanxi and Yu, Shijie},
  journal={Pattern Recognition Letters},
  volume={185},
  pages={38--44},
  year={2024},
  publisher={Elsevier}
}
```