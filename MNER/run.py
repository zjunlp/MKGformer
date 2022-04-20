import os
import argparse
import logging
import sys
sys.path.append("..")

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from models.unimo_model import UnimoCRFModel
from models.modeling_clip import CLIPModel

from processor.datasets import MMPNERBertProcessor, MMPNERBertDataset
from modules.train import BertTrainer
from utils.utils import set_seed

from transformers import BertConfig, CLIPConfig, BertModel, CLIPProcessor

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# from tensorboardX import SummaryWriter

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASS = {
    'bert': (MMPNERBertProcessor, MMPNERBertDataset),
}

DATA_PATH = {

    'twitter17': {'train': 'data/twitter2017/train.txt',
                'dev': 'data/twitter2017/valid.txt',
                'test': 'data/twitter2017/test.txt',
                'predict': 'data/twitter2017/test.txt',
                'train_auximgs': 'data/twitter2017/twitter2017_train_dict.pth',
                'dev_auximgs': 'data/twitter2017/twitter2017_val_dict.pth',
                'test_auximgs': 'data/twitter2017/twitter2017_test_dict.pth',
                'predict_auximgs': 'data/twitter2017/twitter2017_test_dict.pth',
                'img2crop': 'data/twitter17_detect/twitter17_img2crop.pth'},
}

AUX_PATH = {
    'twitter17': {'train': 'data/twitter2017_aux_images/train/crops',
                'dev': 'data/twitter2017_aux_images/val/crops',
                'test': 'data/twitter2017_aux_images/test/crops',
                'predict': 'data/twitter2017_aux_images/test/crops'}
}


IMG_PATH = {
    'twitter17': 'data/twitter2017_images',
}

LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert', type=str, help="model name")
    parser.add_argument('--dataset_name', default='twitter17', type=str, help="The name of dataset.")
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="Pretrained language model name")
    parser.add_argument('--vit_name', default='vit', type=str, help="The name of vit.")
    parser.add_argument('--num_epochs', default=30, type=int, help="Training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=16, type=int, help="batch size")
    parser.add_argument('--lr', default=2e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=16, type=int)
    parser.add_argument('--seed', default=1, type=int, help="random seed, default is 1")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default=None, type=str, help="save model at save_path")
    parser.add_argument('--write_path', default=None, type=str, help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--crf_lr', default=5e-2, type=float, help="learning rate")
    parser.add_argument('--prompt_lr', default=3e-4, type=float, help="learning rate")
    parser.add_argument('--aux_size', default=128, type=int, help="batch size")
    parser.add_argument('--rcnn_size', default=64, type=int, help="batch size")

    args = parser.parse_args()

    data_path, img_path, aux_path = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], AUX_PATH[args.dataset_name]
    data_process, dataset_class = MODEL_CLASS[args.model_name]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    set_seed(args.seed) # set seed, default is 1
    if args.save_path is not None:  # make save_path dir
        args.save_path = os.path.join(args.save_path, args.model_name, args.dataset_name+"_"+str(args.batch_size)+"_"+str(args.lr)+args.notes)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    print(args)
    # logdir = "logs/" + args.model_name+ "_"+args.dataset_name+ "_"+str(args.batch_size) + "_" + str(args.lr) + "simple_bert_test"
    # writer = SummaryWriter(logdir=logdir)
    writer = None
    if args.do_train:
        label_mapping = {label:idx for idx, label in enumerate(LABEL_LIST, 1)}
        label_mapping["PAD"] = 0
        
        clip_vit, clip_processor, aux_processor, rcnn_processor = None, None, None, None
        clip_processor = CLIPProcessor.from_pretrained(args.vit_name)
        aux_processor = CLIPProcessor.from_pretrained(args.vit_name)
        aux_processor.feature_extractor.size, aux_processor.feature_extractor.crop_size = args.aux_size, args.aux_size
        rcnn_processor = CLIPProcessor.from_pretrained(args.vit_name)
        rcnn_processor.feature_extractor.size, rcnn_processor.feature_extractor.crop_size = args.rcnn_size, args.rcnn_size
        clip_model = CLIPModel.from_pretrained(args.vit_name)
        clip_vit = clip_model.vision_model

        processor = data_process(data_path, args.bert_name, clip_processor=clip_processor, aux_processor=aux_processor, rcnn_processor=rcnn_processor)
        train_dataset = dataset_class(processor, label_mapping, transform, img_path, aux_path, max_seq=args.max_seq, ignore_idx=args.ignore_idx, aux_size=args.aux_size, rcnn_size=args.rcnn_size, mode='train')
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        dev_dataset = dataset_class(processor, label_mapping, transform, img_path, aux_path, max_seq=args.max_seq, ignore_idx=args.ignore_idx, aux_size=args.aux_size, rcnn_size=args.rcnn_size, mode='dev')
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,  pin_memory=True)

        test_dataset = dataset_class(processor, label_mapping, transform, img_path, aux_path, max_seq=args.max_seq, ignore_idx=args.ignore_idx, aux_size=args.aux_size, rcnn_size=args.rcnn_size, mode='test')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,  pin_memory=True)  

        # test
        vision_config = CLIPConfig.from_pretrained(args.vit_name).vision_config
        text_config = BertConfig.from_pretrained(args.bert_name)
        model = UnimoCRFModel(LABEL_LIST, args, vision_config, text_config)

        trainer = BertTrainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model, label_map=label_mapping, args=args, logger=logger, writer=writer)
        bert = BertModel.from_pretrained(args.bert_name)
        clip_model_dict = clip_vit.state_dict()
        text_model_dict = bert.state_dict()
        trainer.train(clip_model_dict, text_model_dict)
        torch.cuda.empty_cache()
        # writer.close()

if __name__ == "__main__":
    main()
