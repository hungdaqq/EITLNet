## EFFECTIVE IMAGE TAMPERING LOCALIZATION VIA ENHANCED TRANSFORMER AND CO-ATTENTION FUSION 
### Network Architecture
![EITLNet](./EITLNet.png)

### Update

- 21.04.26, We updated the weight which can be downloaded from [Google Drive Link](https://drive.google.com/drive/my-drive?hl=zh-cn) or [Baiduyun Link](https://pan.baidu.com/s/1ltB8YJO2szg6aXI-IpSOqg)  (password：EITL) and the file `nets/EITLnet.py`. The latest corrected experimental results are marked in <font color=Red>red</font> in the table below, which the average performance is more higher than before([paper](https://ieeexplore.ieee.org/abstract/document/10446332) ).

<img src="./corrected.png" alt="corrected" style="zoom:100%;" />

### Environment

- Python 3.8
- cuda11.1+cudnn8.0.4

### Requirements

- pip install requirements.txt

### Training datasets

The training dataset catalog is as follows. The mask image in the folder has only two values of 0 and 1.

```
├─train_dataset
    ├─ImageSets
    │  └─Segmentation
    │          train.txt
    │          val.txt
    ├─JPEGImages
    │      00001.jpg
    │      00002.jpg
    │      00003.jpg     
    │      ...
    └─SegmentationClass
            00001_gt.png
            00002_gt.png
            00003_gt.png
```

### Trained Models
Please download the weight from [Google Drive Link](https://drive.google.com/drive/my-drive?hl=zh-cn) or [Baiduyun Link](https://pan.baidu.com/s/1ltB8YJO2szg6aXI-IpSOqg)(password:EITL) and place it in the `weights/` directory.

### Training
```python
python train.py
```

### Testing

```
python test.py
```

## Bibtex
 ```
@inproceedings{guo2023effective,
  title={Effective Image Tampering Localization via Enhanced Transformer and Co-attention Fusion},
  author={Guo, Kun and Zhu, Haochen and Cao, Gang},
  booktitle={ICASSP},
  year={2024}
}
 ```
### Contact

If you have any questions, please contact me(guokun21@qq.com).
