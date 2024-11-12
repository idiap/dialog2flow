# -*- coding: utf-8 -*-
"""
Utility functions and SentenceTransformer wrappers for baseline models.

Copyright (c) 2024 Idiap Research Institute
MIT License

@author: Sergio Burdisso (sergio.burdisso@idiap.ch)
"""
import re
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from joblib import Memory
from openai import OpenAI
from datetime import datetime
from tqdm.autonotebook import trange
from sentence_transformers import util
from sentence_transformers.util import batch_to_device
from tenacity import retry, wait_random_exponential, stop_after_attempt
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer, AutoModel


STR_MODEL_COLUMN = "Model"
STR_AVERAGE_COLUMN = "AVG."

memory = Memory('__cache__/chatgpt/labels_action', verbose=0)
memory_events = Memory('__cache__/chatgpt/labels_events', verbose=0)


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_openai_response(client, messages: list, model="gpt-4o", seed=42):
    response = client.chat.completions.create(
        seed=seed,
        model=model,
        messages=messages
    )
    return response.choices[0].message.content


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_openai_embedding(client, docs: list, model="text-embedding-3-large", dimensions=768) -> list[float]:
    if type(docs) == np.ndarray:
        docs = docs.tolist()
    data = client.embeddings.create(input=docs, model=model, dimensions=dimensions).data
    if len(data) > 1:
        return [d.embedding for d in data]
    return data[0].embedding


gpt_client = None
def init_gpt(model_name="gpt-4-turbo-2024-04-09", seed=42):
    global gpt_client, gpt_model, gpt_seed
    if gpt_client is None:
        gpt_client = OpenAI()
    gpt_model = model_name
    gpt_seed = seed


@memory.cache
def get_cluster_label(utterances):
    messages = [
            {"role": "system", "content": """Your task is to annotate conversational utterances with the intent expressed as canonical forms. A canonical form is a short summary representing the intent of a set of utterances - it is neither too verbose nor too short.
Be aware that required canonical forms should avoid containing specific names or quantities, only represent the intent in abstract terms.
For example, for:

For the following utterances:
    1. Uh yes i'm looking for a place for entertainment that is in the center of the city
    2. i would like to know where a place for entertainment that is not far away from my location
Canonical form is: "request entertainment place and inform location"

For the following utterances:
    1. Okay so the phone number is a 1223217297
    2. Sure, my phone number is four four five five
    3. 2 3 4 5 6 is her phone number
Canonical form is: "inform phone number"

For the following utterances:
    1. 8 4 0
    2. yes five five three
Canonical form is: "inform number"

For the following utterances:
    1. I'm just trying to check up on the status of payment. Um, I had requested to reopen this claim. It was approved. Um, I, it was assigned an adjuster and then reassigned an adjuster and then I sent emails to the adjusters, their supervisors and the directors and I still have not been able to get any kind of, uh, status update.
    2. Uh I, I don't understand how it would be closed. We did an inspection, um and we never got any response, any calls, any, anything. So, I mean, I don't understand how it would be closed. They never sent us a field adjuster report, a denial letter. Nothing. That's, that's what I'm saying. I, I this claim was filed in like April and I've never heard anything from Jane whatsoever. I just thought after multiple, multiple emails, I finally got a call from the field adjuster and that was two months ago, we completed that inspection a month ago. And then, you know, it's been crickets ever since I sent her multiple follow ups and there's been absolutely nothing. So yes, Absolutely, I definitely need supplemental. I mean, a coverage decision at this point. To be honest,
    3. Ok. Um here's the problem the adjuster she got that done her last day.
    4. Um I haven't I just haven't um this every time I call or email Miss June um she says she's missing some information from you guys.
Canonical form is: "problem statement"
"""},
            {"role": "user", "content": """Give the following list of utterance provide a single canonical name that represent all of them:
{utts}
""".replace("{utts}", "\n".join(f"{ix+1}. {utt}" for ix, utt in enumerate(utterances)))},
            {"role": "assistant", "content": 'The canonical name that represent the above utterances is: "'}
        ]
    response = get_openai_response(gpt_client,
                                messages,
                                model=gpt_model,
                                seed=gpt_seed)
    m = re.match(r'.+?:\s*"(.+?)".*', response)
    if m:
        response = m.group(1)
    else:
        m = re.match(r'.+?:\s*"(.+)', response)
        if m:
            response = m.group(1)
        else:
            print("Unable to parse response, using raw value:", response)
    return response.strip('"').title()


def slugify(text):
    if "outputs/" in text:
        text = text.split("outputs/")[1]
    return "-".join(re.findall(r"\w+", text))


def get_turn_text(turn: dict, use_ground_truth: bool = False):
    if use_ground_truth:
        # (id, speaker, acts)
        if not turn["turn"] or ":" not in turn["turn"]:
            return "unknown"
        dial_act = turn["turn"].split(": ")[1]
        if re.match(r"^\w+-(\w)", dial_act):
            return dial_act.split("-")[1]
        return dial_act
    return turn["text"]


# https://aclanthology.org/D19-1006.pdf
# https://aclanthology.org/2022.emnlp-main.603.pdf
def compute_anisotropy(embs):
    sim = util.cos_sim(embs, embs)
    sim.fill_diagonal_(0)
    return (sim.sum() / (sim.shape[0] ** 2 - sim.shape[0])).item()


def get_print_column_value(row, column, percentage=False, extra_value=None):
    rank = row[f"{column}_rank"]
    value = row[column]
    if percentage:
        value = f"{value:.2%}"
    else:
        value = f"{value:.3f}"
    extra_value = f"+{extra_value}" if extra_value and extra_value > 0 else extra_value
    return f"{value} ({extra_value})" if extra_value is not None else value


def show_results(models, domains, score_getter, metric_name="",
                  metric_is_ascending=False, print_table=True, sorted=False,
                  percentage=False, value_extra_getter=None, column_value_getter=None):
    rows = []
    columns = [f"{dom}_{metric_name}" for dom in domains]
    for model in models:
        row = {STR_MODEL_COLUMN: model}
        for ix, column in enumerate(columns):
            domain = domains[ix]
            if domain != STR_AVERAGE_COLUMN:
                row[column] = score_getter(model, domain)
            else:
                row[column] = sum(row[col] for col in columns if STR_AVERAGE_COLUMN not in col) / (len(domains) - 1)
        rows.append(row)

    df = pd.DataFrame.from_dict(rows)
    columns = df.columns[1:]
    for column in columns:
        ranking = df[column].sort_values(ascending=metric_is_ascending).tolist()
        ranking = list(dict.fromkeys(ranking).keys())  # Removing duplicates
        df[f"{column}_rank"] = df[column].map(lambda v: ranking.index(v) + 1)

    avg_ranking_column = f"{STR_AVERAGE_COLUMN}_{metric_name}"

    if sorted:
        df.sort_values(by=[avg_ranking_column], ascending=metric_is_ascending, inplace=True)

    if print_table:
        print_table = []
        for _, row in df.iterrows():
            print_row = {STR_MODEL_COLUMN.upper(): row[STR_MODEL_COLUMN]}
            for dom in domains:
                column_extra_value = column_value_getter(row[STR_MODEL_COLUMN], dom)
                col_name = f"{dom} ({column_extra_value})" if column_extra_value else dom
                print_row[col_name] = get_print_column_value(row,
                                                             column=f"{dom}_{metric_name}",
                                                             percentage=percentage,
                                                             extra_value=value_extra_getter(row[STR_MODEL_COLUMN], dom) if value_extra_getter else None)
            print_table.append(print_row)
        print(pd.DataFrame.from_dict(print_table).to_markdown(index=False))

    return df


class SentenceTransformerOpenAI():
    """Simple SentenceTransformer wrapper for OpenAI embedding models"""
    def __init__(self, model_name):
        self.client = OpenAI()
        self.model = model_name

    def __call__(self, features):
        return self.forward(features)

    def forward(self, features):
        embedding = get_openai_embedding(self.client, features, model=self.model)
        return {'sentence_embedding': embedding}

    def encode(self, sentences,
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False):

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences[start_index:start_index+batch_size]
            embeddings = self.forward(sentences_batch)[output_value]
            all_embeddings.extend(embeddings)

        if convert_to_numpy:
            all_embeddings = np.asarray([np.array(emb) for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings


class SentenceTransformerDialoGPT():
    """SentenceTransformer wrapper for DialoGPT model"""
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.device = device

        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.do_lower_case = False
        self.max_seq_length = 64

    def tokenize(self, texts, padding = True):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=padding,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_seq_length,
            )
        )
        return output

    def __call__(self, features):
        return self.forward(features)

    def forward(self, features):
        self.model.eval()

        input_ids, attention_mask = features["input_ids"], features["attention_mask"]

        attention_mask = (input_ids != 50256).long()

        embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(embeddings[0] * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        embeddings = embeddings.to("cpu")

        return {'sentence_embedding': embeddings}
        # return {'sentence_embedding': embeddings.numpy().astype(np.float32)}

    def encode(self, sentences,
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False):
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences[start_index:start_index+batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(features)

                embeddings = out_features[output_value]
                embeddings = embeddings.detach()

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        if convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels


# Adapted by Sergio Burdisso from the original (https://github.com/salesforce/dialog-flow-extraction/blob/main/bert_sbd.py#L34)
# To match SentenceTransformer I/O interface.
class SentenceTransformerSbdBERT(BertPreTrainedModel):
    """SentenceTransformer wrapper for SBD-BERT model."""
    def __init__(self, config, args):
        super(SentenceTransformerSbdBERT, self).__init__(config)
        self.args = args
        self.bert = BertModel(config=config)  # Load pretrained bert
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(config.hidden_size, 3)
        self.activation = nn.Softmax(dim=2)
        self.tokenizer = AutoTokenizer.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
        self.do_lower_case = False
        self.max_seq_length = 64

    def tokenize(self, texts, padding = True):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=padding,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.max_seq_length,
            )
        )
        return output

    def forward(self, features):
        input_ids, attention_mask, token_type_ids = features["input_ids"], features["attention_mask"], features["token_type_ids"]
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )  # last_hidden_state, pooled_output, (hidden_states), (attentions)
        last_hidden_state = outputs[0]
        slot_logits = self.linear(self.dropout(last_hidden_state))
        slot_prob = torch.sum(self.activation(slot_logits)[:, :, 1:],
                              dim=2)  # Add the probs of being B- and I-
        slot_prob = attention_mask * slot_prob
        utt_state = torch.mul(slot_prob.unsqueeze(-1), last_hidden_state)
        utt_state = torch.mean(utt_state, dim=1)

        return {'sentence_embedding': utt_state}

    def encode(self, sentences,
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False):
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences[start_index:start_index+batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(features)

                embeddings = out_features[output_value]
                embeddings = embeddings.detach()

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        # all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def smart_batching_collate(self, batch):
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels
