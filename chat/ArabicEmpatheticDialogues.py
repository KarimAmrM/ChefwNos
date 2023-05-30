# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Arabic Empathetic Dialogues"""

from __future__ import absolute_import, division, print_function

import csv

import datasets
from arabert.preprocess import ArabertPreprocessor

model_name="bert-base-arabert"
arabert_prep = ArabertPreprocessor(model_name=model_name, keep_emojis=False)

_DESCRIPTION = """\
The Arabic Empathetic Dialogues contain ~38K samples of open domain utterances and empathetic responses.
"context" refers to the utterance
"response" refers to the empathetic response
"emotion" refers to the emotion in the utterance
"""

_URL = "https://raw.githubusercontent.com/aub-mind/Arabic-Empathetic-Chatbot/master/arabic-empathetic-conversations.csv"


class ArabicEmpConv(datasets.GeneratorBasedBuilder):
    """ConvAI: A Dataset of Topic-Oriented Human-to-Chatbot Dialogues"""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="myChef",
            version=datasets.Version("1.0.0"),
            description="Full training set",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "Intent": datasets.Value("string"),
                    "Utterance": datasets.Value("string"),
                    "Response": datasets.Value()
                }
            ),
            supervised_keys=None,
            # homepage="https://github.com/aub-mind/Arabic-Empathetic-Chatbot",
        )

    def _split_generators(self, dl_manager):
        downloaded_file = dl_manager.download(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
               # gen_kwargs={"filepath": downloaded_file},
            ),
        ]

    def _generate_examples(self, filepath):
        with open("chat-bot.csv",'r',encoding='utf-8',errors='ignore') as f:
            csv_reader = csv.reader(f)
            for i , row in enumerate(csv_reader):
                if i==0:
                    continue
                yield i-1, {
                    "Intent": arabert_prep.preprocess(row[0]),
                    "Utterance": arabert_prep.preprocess(row[1]),
                    "Response": row[2]
                }