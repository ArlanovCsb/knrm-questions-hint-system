from argparse import ArgumentParser
from collections import Counter, OrderedDict
import logging
import os
import string
import sys
import time

from typing import Dict, List, Tuple, Union, Callable

from flask import Flask, json, request, Response
from langdetect import detect
import nltk
import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
LOG_LEVEL = 'DEBUG'
logger.setLevel(LOG_LEVEL)
logger.addHandler(logging.StreamHandler(sys.stdout))

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

parser = ArgumentParser(description='Get the port')
parser.add_argument('--port', type=str, required=False, default=5000,
                    help='Prediction REST server port')
args, _ = parser.parse_known_args()
PORT = args.port
HOST = '0.0.0.0'


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(
            -0.5 * ((x - self.mu) ** 2) / (self.sigma ** 2)
        )


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: torch.Tensor, freeze_embeddings: bool, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        for i in range(self.kernel_num):
            mu = 1. / (self.kernel_num - 1) + (2. * i) / (
                    self.kernel_num - 1) - 1.0
            sigma = self.sigma
            if mu > 1.0:
                sigma = self.exact_sigma
                mu = 1.0
            kernels.append(GaussianKernel(mu=mu, sigma=sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        out_cont = [self.kernel_num] + self.out_layers + [1]
        mlp = [
            torch.nn.Sequential(
                torch.nn.Linear(in_f, out_f),
                torch.nn.ReLU()
            )
            for in_f, out_f in zip(out_cont, out_cont[1:])
        ]
        mlp[-1] = mlp[-1][:-1]
        return torch.nn.Sequential(*mlp)

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        # shape = [B, L, D]
        embed_query = self.embeddings(query.long())
        # shape = [B, R, D]
        embed_doc = self.embeddings(doc.long())

        # shape = [B, L, R]
        matching_matrix = torch.einsum(
            'bld,brd->blr',
            F.normalize(embed_query, p=2, dim=-1),
            F.normalize(embed_doc, p=2, dim=-1)
        )
        return matching_matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        query, doc = inputs['query'], inputs['document']
        # shape = [B, L, R]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape [B, K]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape [B]
        out = self.mlp(kernels_out)
        return out


class Solution:
    def __init__(self,
                 freeze_knrm_embeddings: bool = True,
                 knrm_kernel_num: int = 21,
                 knrm_out_mlp: List[int] = [],
                 ):
        logger.info("Initialize Solution starting...")
        self.knrm_emb_matrix_path = os.getenv('EMB_PATH_KNRM')
        self.vocab_path = os.getenv('VOCAB_PATH')
        self.mlp_path = os.getenv('MLP_PATH')
        self.glove_vectors_path = os.getenv('EMB_PATH_GLOVE')
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self._load_mlp_model()
        self._load_vocab()
        self._load_glove_vectors()
        self.faiss_index = None
        logger.info("Initialized Solution")

    def _load_mlp_model(self):
        emb_matrix = torch.load(self.knrm_emb_matrix_path)['weight']
        self.emb_dimention = emb_matrix.shape[1]
        self.mlp = KNRM(emb_matrix, self.freeze_knrm_embeddings, self.knrm_kernel_num, out_layers=[])
        mlp_state = torch.load(self.mlp_path)
        logger.debug(f'Embeding dimention is {self.emb_dimention}')
        new_state_dict = OrderedDict()
        new_state_dict['embeddings.weight'] = emb_matrix
        new_state_dict['mlp.0.0.weight'] = mlp_state['0.0.weight']
        new_state_dict['mlp.0.0.bias'] = mlp_state['0.0.bias']
        self.mlp.load_state_dict(new_state_dict)
        logger.info("Loaded MLP object")
        logger.debug(self.mlp.eval())

    def _load_vocab(self):
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        logger.info("Loaded vocab")

    def _load_glove_vectors(self):
        self.embedding_data = {}
        with open(self.glove_vectors_path, 'r', encoding='utf-8') as f:
            for line in f:
                current_line = line.rstrip().split(' ')
                self.embedding_data[current_line[0]] = current_line[1:]
        logger.info("Loaded GLOVE dict")

    @staticmethod
    def handle_punctuation(inp_str: str) -> str:
        inp_str = str(inp_str)
        for punct in string.punctuation:
            inp_str = inp_str.replace(punct, ' ')
        return inp_str

    def simple_preproc(self, inp_str: str):
        base_str = inp_str.strip().lower()
        str_wo_punct = self.handle_punctuation(base_str)
        return nltk.word_tokenize(str_wo_punct)


model = Solution()


@app.route('/ping')
def ping():
    return json.dumps({'status': 'ok'})


@app.route('/query', methods=['POST'])
def query():
    if model.faiss_index is None or not model.faiss_index.is_trained:
        return json.dumps({'status': 'FAISS is not initialized!'})
    data = request.get_json()


@app.route('/update_index', methods=['POST'])
def update_index():
    data = request.get_json()


if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
