import torch
from torch import nn
from util import mean_pooling, convert_numeral_to_six_levels
from model_base import LevelEstimaterBase

torch.set_grad_enabled(True)


class LevelEstimaterClassification(LevelEstimaterBase):
    def __init__(self, corpus_path, test_corpus_path, pretrained_model, problem_type, with_ib, with_loss_weight,
                 attach_wlv, num_labels,
                 word_num_labels, alpha,
                 ib_beta,
                 batch_size,
                 learning_rate,
                 warmup,
                 lm_layer):
        super().__init__(corpus_path, test_corpus_path, pretrained_model, with_ib, attach_wlv, num_labels,
                         word_num_labels, alpha,
                         batch_size,
                         learning_rate, warmup, lm_layer)
        self.save_hyperparameters()

        self.problem_type = problem_type
        self.with_loss_weight = with_loss_weight
        self.ib_beta = ib_beta
        self.dropout = nn.Dropout(0.1)

        self.slv_classifier = nn.Linear(self.lm.config.hidden_size, 1)
        self.loss_fct = nn.MSELoss()

    def forward(self, inputs, return_logits=False):
        # in lightning, forward defines the prediction/inference actions
        outputs, information_loss = self.encode(inputs)
        outputs = mean_pooling(outputs, attention_mask=inputs['attention_mask'])
        logits = self.slv_classifier(self.dropout(outputs))
        if return_logits:
            return logits

        # convert_numeral_to_six_levels returns NP array, need to get it back to a tensor for use later
        predictions = torch.from_numpy(convert_numeral_to_six_levels(logits.detach().clone().cpu().numpy()))
        predictions = predictions.cpu().numpy()

        return predictions
