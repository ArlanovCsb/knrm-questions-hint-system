from argparse import ArgumentParser
from collections import OrderedDict
import logging
import os
import string
import sys

from typing import Dict, List, Tuple, Optional

import faiss
from flask import Flask, json, request, Response, jsonify
from langdetect import detect
import nltk
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
LOG_LEVEL = 'INFO'
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

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
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
                 # index_factory_string='IVF2048_HNSW32,Flat',
                 index_factory_string='IDMap,Flat',
                 faiss_vector_max_len=30,
                 k_ann_num=100,
                 prediction_num=10
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
        self.index_factory_string = index_factory_string
        self.words_text_max_len = faiss_vector_max_len
        self.k_ann_num = k_ann_num
        self.prediction_num = prediction_num
        self.faiss_index = None
        self.docs_database = None
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

    def _preproc_one_document(self, doc):
        p_q = self.simple_preproc(doc)
        word_emb = [self.embedding_data.get(w) for w in p_q[:self.words_text_max_len]
                    if self.embedding_data.get(w) is not None]
        return word_emb

    def _preproc_docs_for_faiss(self) -> Tuple[np.array, np.array]:
        idxs = []
        docs_vectors = []
        for i in self.docs_database.keys():
            d = self.docs_database[i]
            word_emb = self._preproc_one_document(d)
            if len(word_emb) > 0:
                word_emb = np.array(word_emb, dtype=np.float32)
                word_emb = word_emb.max(axis=0)
            else:
                word_emb = np.zeros(self.emb_dimention, dtype=np.float32)
            idxs.append(int(i))
            docs_vectors.append(word_emb)
        logger.info(f"Documents db was prepared. Len is {len(idxs)}")
        return np.array(idxs), np.array(docs_vectors)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        res = [self.vocab.get(i, self.vocab['OOV']) for i in tokenized_text[:self.words_text_max_len]]
        return res

    def _convert_text_to_token_idxs(self, text: str) -> List[int]:
        tokenized_text = self.simple_preproc(text)
        idxs = self._tokenized_text_to_index(tokenized_text)
        return idxs

    def _preproc_data_for_knrm(self, query: str, docs: np.array) -> Dict[str, torch.LongTensor]:
        query = torch.LongTensor([self._convert_text_to_token_idxs(query)] * len(docs))
        max_len = -1
        p_docs = []
        for i in docs:
            if i >= 0:
                d = self._convert_text_to_token_idxs(self.docs_database[str(i)])
                p_docs.append(d)
                max_len = max(len(d), max_len)
        document = [pd + [0] * (max_len - len(pd)) for pd in p_docs]
        document = torch.LongTensor(document)
        result = {'query': query,
                  'document': document}
        return result

    def train_faiss_model(self, raw_docs: Dict[str, str]) -> int:
        self.docs_database = raw_docs
        idxs, docs_vectors = self._preproc_docs_for_faiss()
        index = faiss.index_factory(self.emb_dimention, self.index_factory_string)
        logger.debug('Index train is started...')
        index.train(docs_vectors)
        index.add_with_ids(docs_vectors, idxs)
        logger.debug('Index train has done')
        self.faiss_index = index
        return index.ntotal

    def predict(self, queries: List) -> Tuple[List[bool], List[Optional[List[Tuple[str, str]]]]]:
        lang_check = []
        suggestions = []
        with torch.no_grad():
            self.mlp.eval()
            for i in range(len(queries)):
                lc = False
                sug = None
                if detect(queries[i]) == 'en':
                    word_emb = self._preproc_one_document(queries[i])
                    if len(word_emb) > 0:
                        word_emb = np.array(word_emb, dtype=np.float32)
                        word_emb = word_emb.max(axis=0).reshape(1, -1)
                        _, k_neinborns = self.faiss_index.search(word_emb, self.k_ann_num)
                        k_neinborns = k_neinborns.squeeze()
                        inputs = self._preproc_data_for_knrm(queries[i], k_neinborns)
                        preds = self.mlp(inputs)
                        if len(preds) > 0:
                            preds = np.array(preds.squeeze())
                            logger.debug(f'Query is {queries[i]}')
                            argsort = np.argsort(preds)[::-1]
                            argsort = argsort[:self.prediction_num]
                            pred_idxs = k_neinborns[argsort]
                            pred = [(str(i), self.docs_database[str(i)]) for i in pred_idxs]
                            logger.debug(f'Preds is {pred}')
                            sug = pred
                            lc = True
                lang_check.append(lc)
                suggestions.append(sug)
            return lang_check, suggestions


model = Solution()


@app.route('/ping')
def ping():
    # return jsonify({'status': 'ok'})
    response = json.dumps({'status': 'ok'})
    return Response(response=response, status=200, mimetype="application/json")


@app.route('/query', methods=['POST'])
def query():
    if model.faiss_index is None or not model.faiss_index.is_trained:
        # return jsonify({'status': 'FAISS is not initialized!'})
        response = json.dumps({'status': 'FAISS is not initialized!'})
    else:
        # queries = request.get_json()['queries']
        queries = json.loads(request.json)['queries']
        lang_check, suggestions = model.predict(queries)
        # return jsonify({'lang_check': lang_check, 'suggestions': suggestions})
        response = json.dumps({'lang_check': lang_check, 'suggestions': suggestions})
    return Response(response=response, status=200, mimetype="application/json")


@app.route('/update_index', methods=['POST'])
def update_index():
    # documents = request.get_json()['documents']
    documents = json.loads(request.json)['documents']
    index_size = model.train_faiss_model(documents)
    # return jsonify({'status': 'ok', 'index_size': index_size})
    response = json.dumps({'status': 'ok', 'index_size': index_size})
    return Response(response=response, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
