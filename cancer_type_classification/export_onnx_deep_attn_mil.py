#!/opt/software/install/miniconda38/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

def exists(val):
    return val is not None

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 1024, D = 256, n_tasks = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = nn.Sequential(nn.Linear(L, D), nn.Tanh(), nn.Dropout(0.25))
        self.attention_b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid(), nn.Dropout(0.25))
        self.attention_c = nn.Linear(D, n_tasks)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
TOAD multi-task + concat mil network w/ attention-based pooling
args:
    gate: whether to use gating in attention network
    size_args: size config of attention network
    dropout: whether to use dropout in attention network
    n_classes: number of classes
"""

class TOAD_fc_mtl_concat(nn.Module):

    def __init__(self, input_dim = 1024, size_arg = "big", n_classes = 2):
        super(TOAD_fc_mtl_concat, self).__init__()
        self.size_dict = {"small": [input_dim, 512, 256], "big": [input_dim, 512, 384]}
        size = self.size_dict[size_arg]

        self.attention_net = nn.Sequential(
            nn.Linear(size[0], size[1]), 
            nn.ReLU(), 
            nn.Dropout(0.25),
            Attn_Net_Gated(L = size[1], D = size[2], n_tasks = 1))
        
        self.classifier = nn.Linear(size[1], n_classes)

        initialize_weights(self)
        
    def forward(self, h, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0) 

        attentions = A
        
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h)

        logits  = self.classifier(M)

        return logits, attentions

"""
TOAD multi-modality mil network w/ attention-based pooling
args:
    input_dim: the input dimension
    size_args: size config of attention network
    n_classes: number of classes
"""

class TOAD(nn.Module):

    def __init__(self, input_dim = 1024, size_arg = "big", n_classes = 2):
        super(TOAD, self).__init__()
        self.size_dict = {"small": [input_dim, 512, 256], "big": [input_dim, 512, 384]}
        size = self.size_dict[size_arg]

        self.sex_embedding = nn.Embedding(2, size[1])
        self.age_embedding = nn.Embedding(100, size[1])
        self.origin_embedding = nn.Embedding(2, size[1])

        self.attention_net = nn.Sequential(
            nn.Linear(size[0], size[1]), 
            nn.ReLU(), 
            nn.Dropout(0.25),
            Attn_Net_Gated(L = size[1], D = size[2], n_tasks = 1))
        
        self.classifier = nn.Linear(size[1], n_classes)

        initialize_weights(self)
        
    #def forward(self, h, sex=None, age=None, origin=None, return_features=False, attention_only=False):
    #def forward(self, h, sex, age, origin):
    def forward(self, h):
        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0) 

        attentions = A
        
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h)

        #M = M + self.sex_embedding(sex) + self.age_embedding(age) + self.origin_embedding(origin)

        logits  = self.classifier(M)

        return logits, attentions

if __name__ == '__main__':
    net=TOAD_fc_mtl_concat(384, n_classes=3) 
    path = 'data8/model-reads-1000-patients-trn2308-val300-test800-tiny.pt'
    state_dict=torch.load(path,map_location='cpu')
    msg=net.load_state_dict(state_dict)
    print(msg)

    x=torch.randn(8,384)
    sex=torch.tensor([0])
    origin=torch.tensor([1])
    age=torch.tensor([90])

    net.eval()
    print(net(x))

    traced_script_module = torch.jit.trace(net, x)
    print(traced_script_module(x))

    #traced_script_module.save("traced_cfDNA_methy_attn_mil.pt")

    import onnxruntime as ort
    dynamic_axes={"inputs_embeds": {0:"read_nums"}}
    torch.onnx.export(net, x, "attn_based_deep_mil_tissue_of_origin.onnx", verbose=False, input_names=['inputs_embeds'], output_names=['logits', 'attentions'], dynamic_axes=dynamic_axes)
    ort_session = ort.InferenceSession("attn_based_deep_mil_tissue_of_origin.onnx")
    onnx_outputs = ort_session.run(None, {'inputs_embeds':x.numpy()})
    print(onnx_outputs)

