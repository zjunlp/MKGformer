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

Data Preprocess
==========
To extract visual object images int MNER and MRE tasks, we first use the NLTK parser to extract noun phrases from the text and apply the [visual grouding toolkit](https://github.com/zyang-ur/onestage_grounding) to detect objects. Detailed steps are as follows:

1. Using the NLTK parser (or Spacy, textblob) to extract noun phrases from the text.
2. Applying the [visual grouding toolkit](https://github.com/zyang-ur/onestage_grounding) to detect objects. Taking the twitter2017 dataset as an example, the extracted objects are stored in `twitter2017_aux_images`. The images of the object obey the following naming format: `id_pred_yolo_crop_num.png`, where `id` is the order of the raw image corresponding to the object, `num` is the number of the object predicted by the toolkit. (`id` is doesn't matter.)
3. Establishing the correspondence between the raw images and the objects. We construct a dictionary to record the correspondence between the raw images and the objects. Taking `twitter2017/twitter2017_train_dict.pth` as an example, the format of the dictionary can be seen as follows: `{imgname:['id_pred_yolo_crop_num0.png', 'id_pred_yolo_crop_num1.png', ...] }`, where key is the name of raw images, value is a List of the objects (Note that in `train/val/test.txt`, text and raw image have a one-to-one relationship, so the `imgnae` can be used as a unique identifier for the raw images).

The detected objects and the dictionary of the correspondence between the raw images and the objects are available in our data links.

# Data Download

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

        You can download the image data of FB15k-237 from [mmkb](https://github.com/mniepert/mmkb) which provides a list of image URLs, and refer to more information of description of entity from [kg-bert](https://github.com/yao8839836/kg-bert) repositories.

    + WN18-IMG

        Entity images in WN18 can be obtained from ImageNet, the specific steps can refer to RSME. the [RSME](https://github.com/wangmengsd/RSME) repository.

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
 |    |    |-- twitter2017
 |    |    |    |-- twitter17_detect            # rcnn detected objects
 |    |    |    |-- twitter2017_aux_images      # visual grounding objects
 |    |    |    |-- twitter2017_images          # raw images
 |    |    |    |-- train.txt                   # text data
 |    |    |    |-- ...
 |    |    |    |-- twitter2017_train_dict.pth  # {imgname: [object-image]}
 |    |    |    |-- ...
 |    |-- models        # mner model
 |    |-- modules       # running script
 |    |-- processor     # data process file
 |    |-- utils
 |    |-- run_mner.sh
 |    |-- run.py
 |-- MRE    # Multimodal Relation Extraction
 |    |-- data          # task data
 |    |    |-- img_detect   # rcnn detected objects
 |    |    |-- img_org      # raw images
 |    |    |-- img_vg       # visual grounding objects
 |    |    |-- txt          # text data
 |    |    |    |-- ours_train.txt
 |    |    |    |-- ours_val.txt
 |    |    |    |-- ours_test.txt
 |    |    |    |-- mre_train_dict.pth  # {imgid: [object-image]}
 |    |    |    |-- ...
 |    |    |-- vg_data      # [(id, imgname, noun_phrase)], not useful
 |    |    |-- ours_rel2id.json         # relation data
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
