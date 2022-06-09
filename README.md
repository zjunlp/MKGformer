# MKGFormer

Code for the SIGIR 2022 paper "[Hybrid Transformer with Multi-level Fusion for Multimodal Knowledge Graph Completion](https://arxiv.org/pdf/2205.02357.pdf)"


# Model Architecture

<div align=center>
<img src="resource/model.png" width="75%" height="75%" />
</div>
 
 
 Illustration of MKGformer for (a) Unified Multimodal KGC Framework and (b) Detailed M-Encoder.


# Requirements

To run the codes, you need to install the requirements:
```
pip install -r requirements.txt
```

# Data Collection

The datasets that we used in our experiments are as follows:


+ Twitter2017
    
    You can download the twitter2017 dataset via this link (https://drive.google.com/file/d/1ogfbn-XEYtk9GpUECq1-IwzINnhKGJqy/view?usp=sharing)

    For more information regarding the dataset, please refer to the [UMT](https://github.com/jefferyYu/UMT/) repository.

+ MRE
    
    The MRE dataset comes from [MEGA](https://github.com/thecharm/Mega), many thanks.

    You can download the **MRE dataset with detected visual objects** using folloing command:
    
    ```bash
    cd MRE
    wget 120.27.214.45/Data/re/multimodal/data.tar.gz
    tar -xzvf data.tar.gz
    ```

+ MKG

    + FB15K-237-IMG

        For more information regarding the dataset, please refer to the [mmkb](https://github.com/mniepert/mmkb) and [kg-bert](https://github.com/yao8839836/kg-bert) repositories.

    + WN18-IMG

        For more information regarding the dataset, please refer to the [RSME](https://github.com/wangmengsd/RSME) repository.

The expected structure of files is:


```
MKGFormer
 |-- MKG	# Multimodal Knowledge Graph
 |    |-- dataset       # task data
 |    |-- data          # data process file
 |    |-- lit_models    # lightning model
 |    |-- models        # mkg model
 |    |-- scripts       # running script
 |    |-- main.py   
 |-- MNER	# Multimodal Named Entity Recognition
 |    |-- data          # task data
 |    |-- models        # mner model
 |    |-- modules       # running script
 |    |-- processor     # data process file
 |    |-- utils
 |    |-- run_mner.sh
 |    |-- run.py
 |-- MRE    # Multimodal Relation Extraction
 |    |-- data          # task data
 |    |-- models        # mre model
 |    |-- modules       # running script
 |    |-- processor     # data process file
 |    |-- run_mre.sh
 |    |-- run.py
```

# How to run


+ ## MKG Task

    - First run Image-text Incorporated Entity Modeling to train entity embedding.

    ```shell
        cd MKG
        bash scripts/pretrain_fb15k-237-image.sh
    ```

    - Then do Missing Entity Prediction.


    ```shell
        bash scripts/fb15k-237-image.sh
    ```

+ ## MNER Task

    To run mner task, run this script.

    ```shell
    cd MNER
    bash run_mner.sh
    ```

+ ## MRE Task

    To run mre task, run this script.

    ```shell
    cd MRE
    bash run_mre.sh
    ```

# Acknowledgement

The acquisition of image data for the multimodal link prediction task refer to the code from [https://github.com/wangmengsd/RSME](https://github.com/wangmengsd/RSME), many thanks.

# Papers for the Project & How to Cite
If you use or extend our work, please cite the paper as follows:

```bibtex
@article{DBLP:journals/corr/abs-2205-02357,
  author    = {Xiang Chen and
               Ningyu Zhang and
               Lei Li and
               Shumin Deng and
               Chuanqi Tan and
               Changliang Xu and
               Fei Huang and
               Luo Si and
               Huajun Chen},
  title     = {Hybrid Transformer with Multi-level Fusion for Multimodal Knowledge
               Graph Completion},
  journal   = {CoRR},
  volume    = {abs/2205.02357},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2205.02357},
  doi       = {10.48550/arXiv.2205.02357},
  eprinttype = {arXiv},
  eprint    = {2205.02357},
  timestamp = {Wed, 11 May 2022 17:29:40 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2205-02357.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
