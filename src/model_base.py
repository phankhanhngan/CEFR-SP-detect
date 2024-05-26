import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer, AutoModel

torch.set_grad_enabled(True)


class LevelEstimaterBase(pl.LightningModule):
    def __init__(self, corpus_path, test_corpus_path, pretrained_model, with_ib, attach_wlv,
                 num_labels,
                 word_num_labels, alpha,
                 batch_size,
                 learning_rate, warmup,
                 lm_layer):
        super().__init__()
        self.save_hyperparameters()
        self.CEFR_lvs = 6

        if attach_wlv and with_ib:
            raise Exception('Information bottleneck and word labels cannot be used together!')

        self.corpus_path = corpus_path
        self.test_corpus_path = test_corpus_path
        self.pretrained_model = pretrained_model
        self.with_ib = with_ib
        self.attach_wlv = attach_wlv
        self.num_labels = num_labels
        self.word_num_labels = word_num_labels
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup = warmup
        self.lm_layer = lm_layer

        # Load pre-trained model
        self.load_pretrained_lm()

    def load_pretrained_lm(self):
        if 'roberta' in self.pretrained_model:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model, add_prefix_space=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        self.lm = AutoModel.from_pretrained(self.pretrained_model)

    def encode(self, batch):
        outputs = self.lm(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
        return outputs.hidden_states[self.lm_layer], None

    def forward(self, inputs):
        pass
