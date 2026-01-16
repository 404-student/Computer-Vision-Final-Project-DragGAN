import torch
import torch.nn.functional as F
import sys
import os
import argparse
VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(VIZ_DIR, '..'))
RAFT_CORE = os.path.join(PROJECT_ROOT, 'RAFT', 'core')

if RAFT_CORE not in sys.path:
    sys.path.insert(0, RAFT_CORE)
from raft import RAFT
from utils.utils import InputPadder

class RaftFeatureTracker:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        args = argparse.Namespace()
        args.model = model_path
        args.small = False
        args.mixed_precision = False
        args.alternate_corr = False
        self.model = torch.nn.DataParallel(RAFT(args))
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint, strict=True) 
        self.model = self.model.module
        self.model.to(device)
        self.model.eval()
        print(f"[RaftTracker] 模型加载自 {model_path}，设备 {device}")

    def _feat_to_raft(self, feat_tensor):
        if feat_tensor.size(1) >= 3:
            rgb = feat_tensor[:, :3, :, :]
        else: rgb = feat_tensor.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        # 缩放到 [0, 255]
        flat = rgb.view(rgb.size(1), -1)
        min_vals = torch.quantile(flat, 0.02, dim=1, keepdim=True)[:, :, None]
        max_vals = torch.quantile(flat, 0.98, dim=1, keepdim=True)[:, :, None]
        rgb_clip = torch.clamp(rgb, min_vals, max_vals)
        rgb_normalized = (rgb_clip - min_vals) / (max_vals - min_vals + 1e-8)
        rgb_255 = rgb_normalized * 255.0
        return rgb_255
    
    @torch.no_grad()
    def track(self, feat0, feat, points):
        """
        feat0, feat: [1, C, H, W]
        points (list): [y, x]
        returns: list: [y, x]
        """
        image0 = self._feat_to_raft(feat0)
        image = self._feat_to_raft(feat)

        # 运行 RAFT
        padder = InputPadder(image0.shape)
        image0_pad, image_pad = padder.pad(image0, image)
        flow_low, flow_up = self.model(image0_pad, image_pad, iters=20, test_mode=True)
        flow_up = padder.unpad(flow_up)
        flow = flow_up[0]

        # 更新点坐标
        new_points = []
        h, w = flow.shape[1], flow.shape[2]
        flow_batched = flow.unsqueeze(0)
        for p in points:
            y_orig, x_orig = p[0], p[1]
            y, x = int(round(y_orig)), int(round(x_orig))
            if 0 <= y < h and 0 <= x < w:
                dx = flow[0, y, x].item()
                dy = flow[1, y, x].item()
                new_y = y_orig + dy
                new_x = x_orig + dx
                new_points.append([new_y, new_x])
            else:
                print(f"[RaftTracker Warning] 点 ({y_orig}， {x_orig}) 超出特征图范围 ({h}， {w})。")
                new_points.append([y_orig, x_orig])
        return new_points