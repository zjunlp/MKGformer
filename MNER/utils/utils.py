import torch
import numpy as np
import random
from torch import nn
from collections import OrderedDict


def seq_to_mask(seq_len, max_len):
    """[get attention mask with sequence length]

    Args:
        seq_len ([torch.tensor]): [shape: bsz, each sequence length in a batch]
    """
    max_len = int(max_len) if max_len else seq_len.max().long()
    cast_seq = torch.arange(max_len).expand(seq_len.size(0), -1).to(seq_len)
    mask = cast_seq.lt(seq_len.unsqueeze(1))
    return mask


def set_seed(seed=2021):
    """sets random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def convert_preds_to_outputs(preds, raw_words, mapping, tokenizer):
    """convet model predicitons to BIO outputs

    Args:
        preds ([torch.Tensor]): [prompt model predictions, (bsz x seq_len x labels)]
        raw_words ([List]): [source raw words]
        mapping ([dict]): [map entity labels to <<>>]
        tokenizer : [BartTokenizer]

    Returns:
        [outputs (List)]: [each item length equal to raw_words, BIO format.]
    """
    id2label = list(mapping.keys())
    pred_eos_index = preds.flip(dims=[1]).eq(1).cumsum(dim=1).long()
    preds = preds[:, 1:]
    pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
    pred_seq_len = (pred_seq_len - 2).tolist()

    word_start_index = len(mapping) + 2
    outputs = []
    for i, pred_item in enumerate(preds.tolist()):
        pred_item = pred_item[:pred_seq_len[i]] # single sentence prediction
        pairs, cur_pair = [], []
        if len(pred_item):  # this sentence prediciton= is not null
            for idx in pred_item:
                if idx < word_start_index:  # is entity
                    if len(cur_pair) > 0:
                        # assert word[i] < word[i+1]
                        if all([cur_pair[i] < cur_pair[i + 1] for i in range(len(cur_pair) - 1)]):
                            pairs.append(tuple(cur_pair + [idx]))   # add valid words and current entity id
                    cur_pair = []   # clear word pairs
                else:   # is word
                    cur_pair.append(idx)    # add word id to word pairs
        raw_words_item = raw_words[i]
        cum_lens = [1]
        start_idx = 1
        for word in raw_words_item:
            start_idx += len(tokenizer.tokenize(word, add_prefix_space=True))
            cum_lens.append(start_idx)
        cum_lens.append(start_idx+1)
        output = ['O' for _ in range(len(raw_words_item))]
        # pairs: List[(word id, ... , entity id), (...), ...]
        for pair in pairs:  # (word id, ... , entity id)
            entity = pair[-1]
            words = []
            for word in pair[:-1]:
                if word-word_start_index in cum_lens:
                    words.append(cum_lens.index(word-word_start_index)) 
            if len(words) == 0: continue
            start_idx = words[0]
            end_idx = words[-1]
            output[start_idx] = f'B-{id2label[entity-2]}'
            for _ in range(start_idx+1, end_idx+1):
                output[_] = f'I-{id2label[entity-2]}'
        outputs.append(output)
    return outputs


def write_predictions(path, texts, labels, imgids=None):
    """[write model predictions to path (conll format)]

    Args:
        path ([str]): [save path]
        texts ([List]): [raw texts]
        labels ([List]): [predict labels]
    """
    print(len(texts), len(labels))
    assert len(texts) == len(labels)
    with open(path, "w", encoding="utf-8") as f:
        # f.writelines("-DOCSTART-	O\n\n")
        for i in range(len(texts)):
            if imgids is not None:
                f.writelines("IMGID:{}\n".format(imgids[i]))
            for j in range(len(texts[i])):
                f.writelines("{}\t{}\n".format(texts[i][j], labels[i][j].upper()))
            f.writelines("\n")


def write_bert_predictions(path, labels):
    """[write model predictions to path (conll format)]

    Args:
        path ([str]): [save path]
        labels ([List]): [predict labels]
    """
    with open(path, "w", encoding="utf-8") as f:
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                f.writelines(labels[i][j].upper())
            f.writelines("\n")


def summary(model, *inputs, batch_size=-1, show_input=True):
    '''
    打印模型结构信息
    :param model:
    :param inputs:
    :param batch_size:
    :param show_input:
    :return:
    Example:
        >>> print("model summary info: ")
        >>> for step,batch in enumerate(train_data):
        >>>     summary(self.model,*batch,show_input=True)
        >>>     break
    '''

    def register_hook(module):
        def hook(module, input, output=None):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

            if show_input is False and output is not None:
                if isinstance(output, (list, tuple)):
                    for out in output:
                        if isinstance(out, torch.Tensor):
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out.size())[1:]
                            ][0]
                        else:
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out[0].size())[1:]
                            ][0]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model)):
            if show_input is True:
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)
    model(*inputs)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("-----------------------------------------------------------------------")
    if show_input is True:
        line_new = f"{'Layer (type)':>25}  {'Input Shape':>25} {'Param #':>15}"
    else:
        line_new = f"{'Layer (type)':>25}  {'Output Shape':>25} {'Param #':>15}"
    print(line_new)
    print("=======================================================================")

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        if show_input is True:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["input_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
        else:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )

        total_params += summary[layer]["nb_params"]
        if show_input is True:
            total_output += np.prod(summary[layer]["input_shape"])
        else:
            total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]

        print(line_new)

    print("=======================================================================")
    print(f"Total params: {total_params:0,}")
    print(f"Trainable params: {trainable_params:0,}")
    print(f"Non-trainable params: {(total_params - trainable_params):0,}")
    print("-----------------------------------------------------------------------")



