from syncode import SyncodeLogitsProcessor, Grammar, language_model
from syncode.grammar_mask.grammar_constrainer import *
from typing import Optional, Literal, Union, Callable
from transformers import PreTrainedTokenizer
import torch
from torch import nn

class SyncodeLogitsListWrapper(SyncodeLogitsProcessor):
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 local_forward_batch_size,
                 mode: Literal["original", "grammar_mask", "grammar_strict"] = "grammar_mask"
                 ):
        # Check inputs
        valid_modes = ["original", "grammar_mask", "grammar_strict"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got {mode}")

        grammar = Grammar("python")

        super().__init__(grammar, tokenizer, mode=mode)
        self.logit_processors = [SyncodeLogitsProcessorHelper(self.grammar_engine) for _ in range(local_forward_batch_size)]
        self.input_size = 0

    def get_scores(self):
        rewards = []
        for idx in range(self.input_size):
            rewards.append(self.logit_processors[idx].get_scores())
        rewards = torch.stack(rewards)
        self.grammar_engine.reset()
        return rewards

    def __call__(self, input_idss: torch.LongTensor, scoress: torch.FloatTensor):
        self.input_size = input_idss.size(0)
        for input_ids, scores, syncode in zip(input_idss, scoress, self.logit_processors):
            input_ids = input_ids.unsqueeze(0)
            scores = scores.unsqueeze(0)
            syncode(input_ids, scores)
        return scoress

class SyncodeLogitsProcessorHelper:

    def __init__(self,
                 grammar_engine: GrammarConstrainer
                 ):

        self.do_sample = True
        self.grammar_engine = grammar_engine
        self.reward = []
        self.input_size = None
        self.start_from = 0
        self.generation_started = False
        self.first_generation_finished = False

    def _get_next_token(self, next_token_scores):
        # token selection
        if self.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        return next_tokens
    def reset_scores(self):
        self.reward.clear()
        self.generation_started = False
        self.first_generation_finished = False
        self.input_size = None
        self.start_from = 0

    def set_input_size(self, input_ids):
        if self.input_size is None:
            self.input_size = input_ids.size(1)
            self.start_from = self.input_size

    def code_generation_started(self, input_ids: torch.LongTensor):
        generated_text = input_ids[:, self.input_size:]
        generated_text = self.grammar_engine.tokenizer.batch_decode(generated_text)[0]
        if "```python" in generated_text:
            self.input_size = input_ids.size(1)
            self.start_from = input_ids.size(1) + 1
            self.grammar_engine.start_from = self.start_from
            self.reward[-2:] = [2.0, 2.0]
            return True
        return False

    def code_generation_finished(self, input_ids: torch.LongTensor):
        generated_text = input_ids[:, self.input_size:]
        generated_text = self.grammar_engine.tokenizer.batch_decode(generated_text)[0]
        if "```" in generated_text:
            self.input_size = input_ids.size(1)
            self.start_from = input_ids.size(1)
            self.grammar_engine.start_from = self.start_from
            self.reward[-2:] = [2.0, 2.0]
            #self.first_generation_finished = True
            return True
        return False

    def check_generation_status(self, input_ids: torch.LongTensor):
        self.grammar_engine.start_from = self.start_from
        if not self.generation_started:
            self.generation_started = self.code_generation_started(input_ids)
        else:
           self.generation_started = not self.code_generation_finished(input_ids)

    def got_new_line(self, next_token):
        next_var = self.grammar_engine.tokenizer.decode(next_token)
        if "\n" in next_var:
            return True
        return False

    def get_scores(self):
        rewards = torch.Tensor(self.reward).clone()
        self.reset_scores()
        return rewards

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        self.set_input_size(input_ids)
        if not self.first_generation_finished:
            self.check_generation_status(input_ids)
        if self.generation_started and not self.first_generation_finished:
            next_token = self._get_next_token(scores)
            if not self.grammar_engine.is_valid(input_ids, next_token):
                self.reward.append(-1.0)
            else:
                self.reward.append(1.0)
        else:
            self.reward.append(0.0)

