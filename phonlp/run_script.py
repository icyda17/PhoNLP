# -*- coding: utf-8 -*-
import gdown
import torch
from phonlp.annotate_model import JointModel
from phonlp.models.common import utils as util
from phonlp.models.ner.vocab import MultiVocab
from transformers import AutoConfig, AutoTokenizer
from pathlib import Path
from onnxruntime import InferenceSession
from time import time

def download(save_dir, url="https://public.vinai.io/phonlp.pt"):
    util.ensure_dir(save_dir)
    if save_dir[len(save_dir) - 1] == "/":
        model_file = save_dir + "phonlp.pt"
    else:
        model_file = save_dir + "/phonlp.pt"
    gdown.download(url, model_file)


def load(save_dir="./", tokenizer_config_dir=None, download_flag: bool = False, load_from_local: bool = False, device: int = -1, onnx_phobert: str = None):
    if save_dir[len(save_dir) - 1] == "/":
        model_file = save_dir + "phonlp.pt"
    else:
        model_file = save_dir + "/phonlp.pt"
    print("Loading model from: {}".format(model_file))
    checkpoint = torch.load(model_file, lambda storage, loc: storage)
    args = checkpoint["config"]
    vocab = MultiVocab.load_state_dict(checkpoint["vocab"])
    # check device
    if device >= 0 and torch.cuda.is_available():
        device_use = device
    else:
        device_use = -1
    # load model
    phobert = None  # if not use onnx phobert model
    if load_from_local:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_config_dir, use_fast=False)
        config_phobert = AutoConfig.from_pretrained(
            tokenizer_config_dir, output_hidden_states=True)
        if onnx_phobert is not None:
            if device_use >= 0:
                phobert = InferenceSession(onnx_phobert, providers=[(
                    'CUDAExecutionProvider', {'device_id': device_use})])
            else:
                phobert = InferenceSession(onnx_phobert, providers=[
                                           'CPUExecutionProvider'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args["pretrained_lm"], use_fast=False)
        config_phobert = AutoConfig.from_pretrained(
            args["pretrained_lm"], output_hidden_states=True)
        if download_flag:
            tokenizer.save_pretrained(tokenizer_config_dir)
            config_phobert.save_pretrained(tokenizer_config_dir)

    model = JointModel(args=args, vocab=vocab, config=config_phobert,
                       tokenizer=tokenizer, device_use=device_use, phobert_onnx=phobert)
    if phobert is not None:  # onnx
        own_state = model.state_dict()
        for name, param in checkpoint['model'].items():
            try:
                if 'phobert.' not in name:
                    own_state[name].copy_(param)
                else:
                    continue
            except:
                continue
    else:
        model.load_state_dict(checkpoint["model"], strict=False)
    if device_use < 0:
        model.to(torch.device('cpu'))
    else:
        model.to(torch.device(f"cuda:{device_use}"))
    model.eval()
    return model

if __name__ == "__main__":
    # download("./")
    onnx_phobert = '/thuytt14/NLP/onnx/labs/convert_phonlp/models/model.onnx'
    model = load(save_dir="/thuytt14/NLP/bert_topic/resources/phonlp_models",
                 tokenizer_config_dir='/thuytt14/NLP/bert_topic/resources/phonlp_models', 
                 load_from_local=True, onnx_phobert=None, device=0)
    text = "Tôi tên là Thế_Linh ."
    s = time()
    output = model.annotate(text=text)
    e = time()
    print(f"Time process: {e-s}")
    model.print_out(output)
