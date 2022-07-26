import torch
import torch.nn as nn
import numpy as np
from .base import BaseLitModel
from transformers.optimization import get_linear_schedule_with_warmup
from functools import partial
from .utils import LabelSmoothSoftmaxCEV1
from typing import Callable, Iterable, List

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def decode(output_ids, tokenizer):
    return lmap(str.strip, tokenizer.batch_decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True))

class TransformerLitModel(BaseLitModel):
    def __init__(self, model, args, tokenizer=None, data_config={}):
        super().__init__(model, args)
        self.save_hyperparameters(args)
        if args.bce:
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif args.label_smoothing != 0.0:
            self.loss_fn = LabelSmoothSoftmaxCEV1(lb_smooth=args.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.best_acc = 0
        self.first = True
        self.tokenizer = tokenizer
        self.__dict__.update(data_config)

        # resize the word embedding layer
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.decode = partial(decode, tokenizer=self.tokenizer)

        if args.pretrain:
            # when pretrain, only tune embedding layers
            self._freeze_attention()


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        label = batch.pop("label")
        input_ids = batch['input_ids']
        logits = self.model(**batch, return_dict=True).logits
        
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_logits = logits[torch.arange(bs), mask_idx][:, self.entity_id_st:self.entity_id_ed]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"

        if self.args.bce:
            loss = self.loss_fn(mask_logits, labels)
        else:
            loss = self.loss_fn(mask_logits, label)

        if batch_idx == 0:
            print('\n'.join(self.decode(batch['input_ids'][:4])))
        return loss

    def _eval(self, batch, batch_idx, ):
        labels = batch.pop("labels")
        input_ids = batch['input_ids']
        # single label
        label = batch.pop('label')  # bsz
        logits = self.model(**batch, return_dict=True).logits[:, :, self.entity_id_st:self.entity_id_ed] # bsz, len, entites

        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)    # bsz
        bsz = input_ids.shape[0]
        logits = logits[torch.arange(bsz), mask_idx] # bsz, entites
        # get the entity ranks
        # filter the entity
        assert labels[0][label[0]], "correct ids must in filiter!"
        labels[torch.arange(bsz), label] = 0
        assert logits.shape == labels.shape
        logits += labels * -100 # mask entity

        _, outputs = torch.sort(logits, dim=1, descending=True) # bsz, entities   index
        _, outputs = torch.sort(outputs, dim=1)
        ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1
        return dict(ranks = np.array(ranks))

    def validation_step(self, batch, batch_idx):
        result = self._eval(batch, batch_idx)
        return result

    def validation_epoch_end(self, outputs) -> None:
        ranks = np.concatenate([_['ranks'] for _ in outputs])
        total_ranks = ranks.shape[0]

        if not self.args.pretrain:
            l_ranks = ranks[np.array(list(np.arange(0, total_ranks, 2)))]
            r_ranks = ranks[np.array(list(np.arange(0, total_ranks, 2))) + 1]
            self.log("Eval/lhits10", (l_ranks<=10).mean())
            self.log("Eval/rhits10", (r_ranks<=10).mean())

        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()

        self.log("Eval/hits10", hits10)
        self.log("Eval/hits20", hits20)
        self.log("Eval/hits3", hits3)
        self.log("Eval/hits1", hits1)
        self.log("Eval/mean_rank", ranks.mean())
        self.log("Eval/mrr", (1. / ranks).mean())
        self.log("hits10", hits10, prog_bar=True)
        self.log("hits1", hits1, prog_bar=True)
  

    def test_step(self, batch, batch_idx):
        result = self._eval(batch, batch_idx)
        # self.log("Test/ranks", np.mean(ranks))
        return result

    def test_epoch_end(self, outputs) -> None:
        ranks = np.concatenate([_['ranks'] for _ in outputs])

        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()

       
        self.log("Test/hits10", hits10)
        self.log("Test/hits20", hits20)
        self.log("Test/hits3", hits3)
        self.log("Test/hits1", hits1)
        self.log("Test/mean_rank", ranks.mean())
        self.log("Test/mrr", (1. / ranks).mean())

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * self.args.warm_up_radio, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
    
    def _freeze_attention(self):
        for k, v in self.model.named_parameters():
            if "word" not in k:
                v.requires_grad = False
            else:
                print(k)
    
    def _freaze_word_embedding(self):
        for k, v in self.model.named_parameters():
            if "word" in k:
                print(k)
                v.requires_grad = False

    @staticmethod
    def add_to_argparse(parser):
        parser = BaseLitModel.add_to_argparse(parser)

        parser.add_argument("--label_smoothing", type=float, default=0.1, help="")
        parser.add_argument("--bce", type=int, default=0, help="")
        return parser
