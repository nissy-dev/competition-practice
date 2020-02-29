import torch
import numpy as np
from catalyst.dl import Callback, CallbackOrder
from sklearn.metrics import recall_score


def macro_recall(outputs, targets):
    pred_labels = [np.argmax(out, axis=1) for out in outputs]
    # target_col = ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']
    recall_grapheme = recall_score(pred_labels[0], targets[:, 0], average='macro')
    recall_consonant = recall_score(pred_labels[1], targets[:, 1], average='macro')
    recall_vowel = recall_score(pred_labels[2], targets[:, 2], average='macro')
    scores = [recall_grapheme, recall_consonant, recall_vowel]
    final_score = np.average(scores, weights=[2, 1, 1])
    return final_score


class MacroRecallCallback(Callback):
    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "macro_recall",
            class_names=None,
            ignore_index=None
    ):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        """
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.class_names = class_names
        self.output_key = output_key
        self.input_key = input_key
        self.outputs_for_grapheme = []
        self.outputs_for_consonant = []
        self.outputs_for_vowel = []
        self.targets = []

    def on_loader_start(self, state):
        self.outputs_for_grapheme = []
        self.outputs_for_consonant = []
        self.outputs_for_vowel = []
        self.targets = []

    def on_batch_end(self, state):
        outputs = state.output[self.output_key]
        outputs = [self._to_numpy(out) for out in outputs]
        targets = self._to_numpy(state.input[self.input_key])
        mean_macro_recall = macro_recall(outputs, targets)
        # score for batch
        state.metric_manager.add_batch_value(name=self.prefix, value=mean_macro_recall)
        # save for epoch
        self.outputs_for_grapheme.extend(outputs[0])
        self.outputs_for_consonant.extend(outputs[1])
        self.outputs_for_vowel.extend(outputs[2])
        self.targets.extend(targets)

    def on_loader_end(self, state):
        outputs = [self.outputs_for_grapheme, self.outputs_for_consonant, self.outputs_for_vowel]
        targets = np.array(self.targets)
        mean_macro_recall = macro_recall(outputs, targets)
        state.metrics.epoch_values[state.loader_name][self.prefix] = mean_macro_recall

    def _to_numpy(self, x):
        """
        Convert whatever to numpy array
        :param x: List, tuple, PyTorch tensor or numpy array
        :return: Numpy array
        """
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, (list, tuple, int, float)):
            return np.array(x)
        else:
            raise ValueError("Unsupported type")
