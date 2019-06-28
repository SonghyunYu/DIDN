import torch
from color_model import _NetG

model1_path = "./checkpoint/pretrained_color/model_31.82db_11ep_32000it_.pth"
model2_path = "./checkpoint/pretrained_color/model_31.85db_17ep_20000it_.pth"

model1 = torch.load(model1_path, map_location=lambda storage, loc: storage)["model"]
model2 = torch.load(model2_path, map_location=lambda storage, loc: storage)["model"]

beta = 0.5 #The interpolation parameter
params1 = model1.named_parameters()
params2 = model2.named_parameters()

dict_params2 = dict(params2)

for name1, param1 in params1:
    if name1 in dict_params2:
        dict_params2[name1].data.copy_(beta*param1.data + (beta)*dict_params2[name1].data)

model = _NetG()
model.load_state_dict(dict_params2)

model_out_path = "checkpoint/" + "color_model.pth"
state = {"epoch": 0, "model": model}
torch.save(state, model_out_path)
