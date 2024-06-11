import sys
import torch
import torch.nn as nn
import onnxruntime as ort
sys.path.append('/data/ai/WSI/data/code/transformers-main')
from src.transformers.models.opt import OPTModel, OPTForSequenceClassification
from src.transformers import AutoTokenizer
sys.path.append("/opt/software/github/byt5-bio/src/")
from count_parameters import *


tokenizer = AutoTokenizer.from_pretrained('/opt/software/github/byt5-bio/opt-alphabet/')

#model = OPTModel.from_pretrained('model2')
model = OPTForSequenceClassification.from_pretrained('model2')
model.eval()

count_parameters(model)

ss = 'A C G T A C G'

inputs = tokenizer(ss, return_token_type_ids=False, return_tensors='pt')

with torch.no_grad():
    ## OPTModel
    #last_hidden_state, attentions, last_hidden_state_mean = model(**inputs)
    ## OPTForSequenceClassification
    logits, last_hidden_state, attentions, last_hidden_state_mean = model(**inputs)

#print(last_hidden_state)
print(logits)


#### Export ONNX model ####
print(inputs['input_ids'].tolist())
dynamic_axes={"input_ids": {1:"seq_length"}}
## OPTModel
#output_names = ['last_hidden_state', 'attentions', 'last_hidden_state_mean']
## OPTForSequenceClassification
output_names = ['logits', 'last_hidden_state', 'attentions', 'last_hidden_state_mean']
torch.onnx.export(model, inputs['input_ids'], "model2/pytorch_model.onnx", verbose=False, input_names=['input_ids'], output_names=output_names, dynamic_axes=dynamic_axes)
ort_session = ort.InferenceSession("model2/pytorch_model.onnx")
onnx_outputs = ort_session.run(None, {'input_ids':inputs['input_ids'].numpy()})
print(['onnx logit:', onnx_outputs[0]])



#from transformers import OPTModel as OPTModelOrigin
from transformers import OPTForSequenceClassification as OPTModelOrigin

modelOrigin = OPTModelOrigin.from_pretrained('model2')
modelOrigin.eval()

with torch.no_grad():
    out = modelOrigin(**inputs, output_attentions=True, output_hidden_states=True)

#print(out.hidden_states)
#print(out.last_hidden_state)

#flag = (last_hidden_state == out.last_hidden_state).all()
#print(['last_hidden_state', flag])

flag = (attentions[0] == out.attentions[0]).all()
print(['attentions', flag])
flag = (out.logits == logits)
print(['logits', flag])
print(['logit:', out.logits])

#flag = (last_hidden_state[-1] == out.last_hidden_state[-1]).all()
#print(['last_hidden_state[-1]', flag])

#flag = (last_hidden_state_mean == out.last_hidden_state.mean(1)).all()
#print(['last_hidden_state_mean', flag])
