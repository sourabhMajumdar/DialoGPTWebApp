"""
    This file is responsible for creating an inference engine
    based on DialoGPT bot
"""

# Importing relevant libraries
import glob
import json
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.notebook import tqdm, trange

from pathlib import Path

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


# Configs
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)



class GPTBot(object):
    def __init__(self) :
        #super().__init__(self)
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
        self.model = AutoModelWithLMHead.from_pretrained('output')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.dialog_history = []
    
    def generateResponse(self,input_text) :

        story = ' {} '.format(self.tokenizer.eos_token).join(self.dialog_history)
        input_ids = self.tokenizer.encode(story + self.tokenizer.eos_token + input_text + self.tokenizer.eos_token,return_tensors='pt')
        input_ids = input_ids.to(self.device)

        # generate a response
        chat_history_ids = self.model.generate(
            input_ids,
            max_length=1000,
            pad_token_id = self.tokenizer.eos_token_id,
            top_p=0.92,
            top_k=50
        )

        bot_response = self.tokenizer.decode(chat_history_ids[:,input_ids.shape[-1]:][0],skip_special_tokens=True)
        self.dialog_history.append(input_text)
        self.dialog_history.append(bot_response)
        return bot_response
    
    def purge_history(self) :
        self.dialog_history = []
