import numpy as np
from catalyst.dl import Callback, CallbackOrder
from sklearn.metrics import recall_score


def macro_recall(outputs, targets):
    pred_labels = [np.argmax(out.detach().cpu().numpy(), axis=1) for out in outputs]
    targets = targets.detach().cpu().numpy()
    # target_col = ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']
    recall_grapheme = recall_score(pred_labels[0], targets[:, 0], average='macro')
    recall_consonant = recall_score(pred_labels[1], targets[:, 1], average='macro')
    recall_vowel = recall_score(pred_labels[2], targets[:, 2], average='macro')
    scores = [recall_grapheme, recall_consonant, recall_vowel]
    final_score = np.average(scores, weights=[2, 1, 1])
    print(f'recall: grapheme {recall_grapheme}, consonant {recall_consonant}, vowel {recall_vowel}, '
          f'total {final_score}, y {targets.shape}')
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

    def on_batch_end(self, state):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]
        mean_macro_recall = macro_recall(outputs, targets)
        state.metric_manager.add_batch_value(name=self.prefix, value=mean_macro_recall)
