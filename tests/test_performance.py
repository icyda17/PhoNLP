import phonlp
from timeit import timeit

def testbatch():
    global nlpmdl, s
    tokens, postags, _, _ = nlpmdl.annotate(s, batch_size=50)
    return tokens, postags

def main():
    from vncorenlp import VnCoreNLP
    rdrsegmenter = VnCoreNLP('/thuytt14/NLP/bert_topic/resources/vncore_models/VnCoreNLP-1.1.1.jar', port=9005)
    text = ["bầu trời ngày hôm nay trong xanh."]*1000 
    
    s = [' '.join(rdrsegmenter.tokenize(i)[0]) for i in text]
    # input can be either string or list. `batch_size` is the only parameter affecting performance
    tokens, postags, _, _ = nlpmdl.annotate(s, batch_size=50)
    return tokens, postags
    

if __name__=='__main__':
    nlpmdl = phonlp.load(save_dir='/thuytt14/NLP/bert_topic/resources/phonlp_models')
    s = ['bầu_trời ngày hôm_nay trong xanh .']*1000
    o1 = timeit(testbatch, number=1000)
    print(o1)