import os
import torch
import random
import transformers
from PIL import Image
from enum import Enum
from os import listdir
from dataclasses import dataclass
from typing import Any, Optional, Union
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer
from transformers.models.clip import CLIPProcessor
from transformers.tokenization_utils_base import (BatchEncoding,
                                                  PreTrainedTokenizerBase)
from .base_data_module import BaseDataModule
from .processor import KGProcessor, get_dataset

transformers.logging.set_verbosity_error()


aux_size, rcnn_size = 128, 64
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
aux_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
aux_processor.feature_extractor.size, aux_processor.feature_extractor.crop_size = aux_size, aux_size
rcnn_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
rcnn_processor.feature_extractor.size, rcnn_processor.feature_extractor.crop_size = rcnn_size, rcnn_size


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the ``padding`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for tab-completion
    in an IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    num_labels: int = 0
    task_name: str = None
    entity_img_path: str = None
    entity_img_files: Optional[Any] = None

    def __call__(self, features, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature.pop("labels") for feature in features] if "labels" in features[0].keys() else None
        label = [feature.pop("label") for feature in features]
        features_keys = {}
        entities = [feature.pop("entity") for feature in features] if "entity" in features[0].keys() else None
        for k in features[0].keys():
            # ignore the padding arguments
            if k in ["input_ids", "attention_mask", "token_type_ids"]: continue
            features_keys[k] = [feature.pop(k) for feature in features]

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        bsz = len(labels)
        with torch.no_grad():
            new_labels = torch.zeros(bsz, self.num_labels)
            for i,l in enumerate(labels):
                if isinstance(l, int): 
                    new_labels[i][l] = 1
                else:
                    for j in l:
                        new_labels[i][j] = 1
            labels = new_labels

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        features['labels'] = labels
        features['label'] = torch.tensor(label)
        features.update(features_keys)

        # region
        pixel_images, aux_images, rcnn_images = [], [], []
        for entity in entities:
            if self.task_name == 'wn18':
                en_file = 'n' + entity    # wn18
            elif self.task_name == 'fb15k-237':
                en_file = entity[1:].replace('/', '.') # m.01rng // n01443537
            else:
                raise ValueError(
                f"{self.task_name} is not a valid task name, please select one of [wn18, fb15k-237]"
            )
            en_imgs = []
            if en_file in self.entity_img_files:
                en_file = os.path.join(self.entity_img_path, en_file)
                en_imgs = [os.path.join(en_file, file) for file in os.listdir(en_file)]
                if len(en_imgs) > 7:    # random select six imgs
                    random.seed(1)
                    en_imgs = random.sample(en_imgs, k=7)
            en_full_imgs = en_imgs[:1]
            en_aux_imgs = en_imgs[1:4]
            en_rcnn_imgs = en_imgs[4:]

            if len(en_full_imgs) > 0:
                full_img = Image.open(en_full_imgs[0]).convert('RGB')
                full_img = clip_processor(images=full_img, return_tensors='pt')['pixel_values'].squeeze()
                pixel_images.append(full_img)
            else:
                pixel_images.append(torch.zeros((3, 224, 224)))

            aux_imgs, rcnn_imgs = [], []
            # select 3 imgs
            for i in range(min(3, len(en_aux_imgs))):
                aux_img = Image.open(en_aux_imgs[i]).convert('RGB')
                aux_img = aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
                aux_imgs.append(aux_img)
            for i in range(min(3, len(en_rcnn_imgs))):
                rcnn_img = Image.open(en_rcnn_imgs[i]).convert('RGB')
                rcnn_img = rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                rcnn_imgs.append(rcnn_img)
            # padding
            for i in range(3-len(en_aux_imgs)):
                aux_imgs.append(torch.zeros((3, aux_size, aux_size))) 
            for i in range(3-len(en_rcnn_imgs)):
                rcnn_imgs.append(torch.zeros((3, rcnn_size, rcnn_size)))
            aux_images.append(torch.stack(aux_imgs))
            rcnn_images.append(torch.stack(rcnn_imgs))

        features['pixel_values'] = torch.stack(pixel_images)
        features['aux_values'] = torch.stack(aux_images)
        features['rcnn_values'] = torch.stack(rcnn_images)
        #endregion
        return features


class KGC(BaseDataModule):
    def __init__(self, args, model) -> None:
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=False)
        self.processor = KGProcessor(self.tokenizer, args)
        self.label_list = self.processor.get_labels(args.data_dir)

        entity_list = self.processor.get_entities(args.data_dir)
        print(len(entity_list))
        
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': entity_list})
        
        entity_img_path = {'wn18': 'dataset/wn18-images/', 'fb15k-237': 'dataset/FB15k-images/'}[self.args.task_name]
        entity_img_files = listdir(entity_img_path)
        self.sampler = DataCollatorForSeq2Seq(self.tokenizer,
            model=model,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if self.args.precision == 16 else None,
            padding="longest",
            max_length=self.args.max_seq_length,
            num_labels=len(entity_list),
            task_name=self.args.task_name,
            entity_img_path=entity_img_path,
            entity_img_files=entity_img_files

        )
        relations_tokens = self.processor.get_relations(args.data_dir)
        self.num_relations = len(relations_tokens)
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': relations_tokens})

        vocab = self.tokenizer.get_added_vocab()    # dict: word: idx
        self.relation_id_st = vocab[relations_tokens[0]]
        self.relation_id_ed = vocab[relations_tokens[-1]] + 1
        self.entity_id_st = vocab[entity_list[0]]
        self.entity_id_ed = vocab[entity_list[-1]] + 1


    def setup(self, stage=None):
        self.data_train = get_dataset(self.args, self.processor, self.label_list, self.tokenizer, "train")
        self.data_val = get_dataset(self.args, self.processor, self.label_list, self.tokenizer, "dev")
        self.data_test = get_dataset(self.args, self.processor, self.label_list, self.tokenizer, "test")

    def prepare_data(self):
        pass

    def get_config(self):
        d = {}
        for k, v in self.__dict__.items():
            if "st" in k or "ed" in k:
                d.update({k:v})
        
        return d

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--data_dir", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--warm_up_radio", type=float, default=0.1, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        return parser

    def get_tokenizer(self):
        return self.tokenizer

    def train_dataloader(self):
        return DataLoader(self.data_train, num_workers=self.num_workers, pin_memory=False, collate_fn=self.sampler, batch_size=self.args.batch_size, shuffle=self.args.pretrain)

    def val_dataloader(self):
        return DataLoader(self.data_val, num_workers=self.num_workers, pin_memory=False, collate_fn=self.sampler, batch_size=self.args.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, num_workers=self.num_workers, pin_memory=False, collate_fn=self.sampler, batch_size=self.args.eval_batch_size)

