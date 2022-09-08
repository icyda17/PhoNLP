# PhoNLP: A BERT-based multi-task learning model for part-of-speech tagging, named entity recognition and dependency parsing


### Installation
```commandline
git clone https://github.com/icyda17/PhoNLP.git
cd PhoNLP
pip3 install -e .
```

### Pre-trained PhoNLP model for Vietnamese
The pre-trained PhoNLP model for Vietnamese can be manually downloaded from https://public.vinai.io/phonlp.pt

### Example usage

Run POS Tag: [function main](tests/test_performance.py)<br><br>
Python API:
```python
import phonlp

# Load the trained PhoNLP model
model = load(save_dir="/absolute/path/to/phonlp_tmp",
            tokenizer_config_dir='/absolute/path/to/phonlp_tmp', 
            load_from_local=True, onnx_phobert="/absolute/path/to/phobert_onnx_model", device='cpu') 
            # onnx_phobert = None if do not use onnx to run model

# Annotate a corpus where each line represents a word-segmented sentence

model.annotate(input_file='/absolute/path/to/input.txt', output_file='/absolute/path/to/output.txt')

# Annotate a word-segmented sentence
model.print_out(model.annotate(text="Xin_chào Hà_Nội", batch_size=2))
```
Detailed information refers to [run_script](phonlp/run_script.py)