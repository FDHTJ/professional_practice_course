import json
import pickle

import torch
import torch.nn.functional as F
with open("img2embedding.pkl", "rb") as f:
    img2embedding = pickle.load(f)
pos_d,neg_d=[],[]
with open("test_triple.json",'r') as f:
    data=json.load(f)
    for d in data:
        pos_d.append(F.pairwise_distance(torch.tensor(img2embedding[d['anchor']]),torch.tensor(img2embedding[d['positive']])))
        neg_d.append(F.pairwise_distance(torch.tensor(img2embedding[d['anchor']]),torch.tensor(img2embedding[d['negative']])))
print( sum(pos_d) / len(pos_d))
print( sum(neg_d) / len(neg_d))

