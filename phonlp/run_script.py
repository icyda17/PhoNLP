# -*- coding: utf-8 -*-
import gdown
import torch
from phonlp.annotate_model import JointModel
from phonlp.models.common import utils as util
from phonlp.models.ner.vocab import MultiVocab
from transformers import AutoConfig, AutoTokenizer
from pathlib import Path

def download(save_dir, url="https://public.vinai.io/phonlp.pt"):
    util.ensure_dir(save_dir)
    if save_dir[len(save_dir) - 1] == "/":
        model_file = save_dir + "phonlp.pt"
    else:
        model_file = save_dir + "/phonlp.pt"
    gdown.download(url, model_file)


def load(save_dir="./", tokenizer_config_dir=None, download_flag: bool = False, load_from_local: bool = False):
    if save_dir[len(save_dir) - 1] == "/":
        model_file = save_dir + "phonlp.pt"
    else:
        model_file = save_dir + "/phonlp.pt"
    print("Loading model from: {}".format(model_file))
    checkpoint = torch.load(model_file, lambda storage, loc: storage)
    args = checkpoint["config"]
    vocab = MultiVocab.load_state_dict(checkpoint["vocab"])
    # load model
    if load_from_local:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config_dir, use_fast=False)
        config_phobert = AutoConfig.from_pretrained(tokenizer_config_dir, output_hidden_states=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args["pretrained_lm"], use_fast=False)
        config_phobert = AutoConfig.from_pretrained(args["pretrained_lm"], output_hidden_states=True) 
        if download_flag:
            tokenizer.save_pretrained(tokenizer_config_dir)
            config_phobert.save_pretrained(tokenizer_config_dir)       

    model = JointModel(args, vocab, config_phobert, tokenizer)
    model.load_state_dict(checkpoint["model"], strict=False)
    if torch.cuda.is_available() is False:
        model.to(torch.device("cpu"))
    else:
        model.to(torch.device("cuda"))
    model.eval()
    return model


if __name__ == "__main__":
    # download("./")
    model = load(local_dir="ab", save_dir="/thuytt14/NLP/bert_topic/resources/phonlp_models")
    # text = "Tôi tên là Thế_Linh ."
    # output = model.annotate(text=text)
    # model.print_out(output)
