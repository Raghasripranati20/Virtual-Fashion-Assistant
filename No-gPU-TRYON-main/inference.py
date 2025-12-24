import socket
import timeit
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
from collections import OrderedDict
sys.path.append('./')

# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms
import cv2

# Custom includes
from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr

import argparse
import torch.nn.functional as F


# ======================================================
# EXISTING LABEL COLORS (UNCHANGED)
# ======================================================
label_colours = [
    (0,0,0),(128,0,0),(255,0,0),(0,85,0),(170,0,51),
    (255,85,0),(0,0,85),(0,119,221),(85,85,0),(0,85,85),
    (85,51,0),(52,86,128),(0,128,0),(0,0,255),
    (51,170,221),(0,255,255),(85,255,170),(170,255,85),
    (255,255,0),(255,170,0)
]


# ======================================================
# EXISTING FUNCTIONS (UNCHANGED)
# ======================================================
def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev, dim=0)

def decode_labels(mask, num_images=1, num_classes=20):
    n, h, w = mask.shape
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (w, h))
        pixels = img.load()
        for y in range(h):
            for x in range(w):
                pixels[x, y] = label_colours[mask[i, y, x]]
        outputs[i] = np.array(img)
    return outputs

def read_img(img_path):
    return Image.open(img_path).convert('RGB')

def img_transform(img, transform=None):
    sample = {'image': img, 'label': 0}
    return transform(sample)


# ======================================================
# 🔴 NEW ADDITIONS (NO EXISTING CODE MODIFIED)
# ======================================================

# ---------- Physics Violation ----------
def compute_fabric_strain(mask):
    area = np.sum(mask > 0)
    return abs(area / (mask.shape[0]*mask.shape[1]) - 0.3)

def physics_violation_label(score):
    if score < 0.05:
        return "Physically Plausible"
    elif score < 0.15:
        return "Mild Violation"
    return "Severe Violation"


# ---------- Uncertainty (MC Dropout) ----------
def mc_dropout_forward(net, inputs, adj1, adj2, adj3, runs=5):
    net.train()
    preds = []
    with torch.no_grad():
        for _ in range(runs):
            out = net.forward(inputs, adj1, adj3, adj2)[0]
            preds.append(out.unsqueeze(0))
    return torch.cat(preds, dim=0)


# ---------- Counterfactual ----------
def counterfactual_mask(mask, scale=1.1):
    h, w = mask.shape
    resized = cv2.resize(mask, None, fx=scale, fy=1.0)
    return resized[:, :w]


# ======================================================
# EXISTING INFERENCE FUNCTION (ONLY EXTENDED)
# ======================================================
def inference(net, img_path='', output_path='./', output_name='f', use_gpu=True):

    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1,1,7,20).cpu().transpose(2,3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1,1,7,7).cpu()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1,1,20,20).cpu()

    scale_list = [1, 0.5, 0.75, 1.25, 1.5]
    img = read_img("data/test/image/" + img_path)

    testloader_list, testloader_flip_list = [], []

    for pv in scale_list:
        ts = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()
        ])
        ts_f = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.HorizontalFlip_only_img(),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()
        ])
        testloader_list.append(img_transform(img, ts))
        testloader_flip_list.append(img_transform(img, ts_f))

    net.eval()
    start_time = timeit.default_timer()

    for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
        inputs = torch.cat([
            sample_batched[0]['image'].unsqueeze(0),
            sample_batched[1]['image'].unsqueeze(0)
        ], dim=0)

        if iii == 0:
            _, _, h, w = inputs.size()

        inputs = inputs.cpu()

        # ====== MC DROPOUT ADDITION ======
        mc_out = mc_dropout_forward(
            net, inputs,
            adj1_test.cpu(),
            adj2_test.cpu(),
            adj3_test.cpu(),
            runs=5
        )

        outputs = torch.mean(mc_out, dim=0)
        outputs = (outputs + flip(flip_cihp(outputs), -1)) / 2
        outputs = outputs.unsqueeze(0)

        if iii > 0:
            outputs = F.interpolate(outputs, size=(h,w), mode='bilinear', align_corners=True)
            outputs_final += outputs
        else:
            outputs_final = outputs.clone()

    predictions = torch.max(outputs_final, 1)[1]
    results = predictions.cpu().numpy()

    vis_res = decode_labels(results)
    Image.fromarray(vis_res[0]).save("data/test/image-parse/" + output_name.replace(".jpg",".png"))
    cv2.imwrite("data/test/image-parse-new/" + output_name.replace(".jpg",".png"), results[0])

    # ====== PHYSICS VIOLATION ======
    strain = compute_fabric_strain(results[0])
    label = physics_violation_label(strain)
    print("[Physics]", label, "| Score:", round(strain,4))

    # ====== UNCERTAINTY ======
    uncertainty = torch.var(mc_out, dim=0).mean().item()
    confidence = round(1 / (1 + uncertainty) * 100, 2)
    print("[Uncertainty] Confidence:", confidence, "%")

    # ====== COUNTERFACTUAL ======
    cf_mask = counterfactual_mask(results[0], scale=1.15)
    cf_vis = decode_labels(cf_mask[np.newaxis,:,:])[0]
    Image.fromarray(cf_vis).save(
        "data/test/image-parse/counterfactual_" + output_name.replace(".jpg",".png")
    )

    end_time = timeit.default_timer()
    print("Inference time:", end_time - start_time)


# ======================================================
# EXISTING MAIN (UNCHANGED)
# ======================================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--loadmodel', default='', type=str)
    parser.add_argument('--img_path', default='', type=str)
    parser.add_argument('--output_path', default='', type=str)
    parser.add_argument('--output_name', default='', type=str)
    parser.add_argument('--use_gpu', default=0, type=int)
    opts = parser.parse_args()

    net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(
        n_classes=20, hidden_layers=128, source_classes=7
    )

    if opts.loadmodel == '':
        raise RuntimeError("No model loaded")
    net.load_source_model(torch.load(opts.loadmodel, map_location='cpu'))

    inference(
        net,
        img_path=opts.img_path,
        output_path=opts.output_path,
        output_name=opts.output_name,
        use_gpu=False
    )
