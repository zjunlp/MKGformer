import sys
sys.path.append("..")

from torchcrf import CRF
from torch import nn
from .modeling_unimo import UnimoModel
from transformers.modeling_outputs import TokenClassifierOutput

class UnimoCRFModel(nn.Module):
    def __init__(self, label_list, args, vision_config, text_config):
        super(UnimoCRFModel, self).__init__()
        self.args = args
        print(vision_config)
        print(text_config)
        self.vision_config = vision_config
        self.text_config = text_config
        self.model = UnimoModel(vision_config, text_config)

        self.num_labels  = len(label_list) + 1  # pad
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.text_config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.batch_id = 0

    def forward(
            self, 
            input_ids=None, 
            attention_mask=None, 
            token_type_ids=None, 
            labels=None, 
            images=None, 
            aux_imgs=None,
            rcnn_imgs=None,
    ):
        bsz = input_ids.size(0)

        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,

                            pixel_values=images,
                            aux_values=aux_imgs, 
                            rcnn_values=rcnn_imgs,
                            return_dict=True,)

        sequence_output = output.last_hidden_state       # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        emissions = self.fc(sequence_output)             # bsz, len, labels
        
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')  # 去掉CLS
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )
