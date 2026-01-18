# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from socket import has_dualstack_ipv6
import sys
import copy
import traceback
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm
import dnnlib
from torch_utils.ops import upfirdn2d
import legacy # pylint: disable=import-error

from torchvision.models import inception_v3
import os
import json

#----------------------------------------------------------------------------

class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            if isinstance(value, CapturedException):
                msg = str(value)
            else:
                msg = traceback.format_exc()
        assert isinstance(msg, str)
        super().__init__(msg)

#----------------------------------------------------------------------------

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out

#----------------------------------------------------------------------------

def add_watermark_np(input_image_array, watermark_text="AI Generated"):
    image = Image.fromarray(np.uint8(input_image_array)).convert("RGBA")

    # Initialize text image
    txt = Image.new('RGBA', image.size, (255, 255, 255, 0))
    font = ImageFont.truetype('arial.ttf', round(25/512*image.size[0]))
    d = ImageDraw.Draw(txt)

    text_width, text_height = font.getsize(watermark_text)
    text_position = (image.size[0] - text_width - 10, image.size[1] - text_height - 10)
    text_color = (255, 255, 255, 128)  # white color with the alpha channel set to semi-transparent

    # Draw the text onto the text canvas
    d.text(text_position, watermark_text, font=font, fill=text_color)

    # Combine the image with the watermark
    watermarked = Image.alpha_composite(image, txt)
    watermarked_array = np.array(watermarked)
    return watermarked_array

def tensor_to_pil(img: torch.Tensor):
    """
    img: torch.Tensor [1, 3, H, W], range [-1, 1]
    return: PIL.Image.Image
    """
    assert isinstance(img, torch.Tensor)
    assert img.ndim == 4 and img.shape[1] == 3

    img = img[0]
    img = (img * 127.5 + 128).clamp(0, 255)
    img = img.to(torch.uint8).permute(1, 2, 0)

    from PIL import Image
    return Image.fromarray(img.cpu().numpy())

#----------------------------------------------------------------------------

class Renderer:
    def __init__(self, disable_timing=False):
        self._device        = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self._dtype         = torch.float32 if self._device.type == 'mps' else torch.float64
        self._pkl_data      = dict()    # {pkl: dict | CapturedException, ...}
        self._networks      = dict()    # {cache_key: torch.nn.Module, ...}
        self._pinned_bufs   = dict()    # {(shape, dtype): torch.Tensor, ...}
        self._cmaps         = dict()    # {name: torch.Tensor, ...}
        self._is_timing     = False
        if not disable_timing:
            self._start_event   = torch.cuda.Event(enable_timing=True)
            self._end_event     = torch.cuda.Event(enable_timing=True)
        self._disable_timing = disable_timing
        self._net_layers    = dict()    # {cache_key: [dnnlib.EasyDict, ...], ...}
        try:
            from .raft_tracker import RaftFeatureTracker
            raft_model_path = './RAFT/models/raft-things.pth'
            self.raft_tracker = RaftFeatureTracker(raft_model_path, self._device)
            print("[Renderer] RAFT跟踪器初始化成功。")
        except Exception as e:
            print(f"[Renderer] RAFT跟踪器初始化失败: {e}")
            self.raft_tracker = None
        self._feat_drag_start = None
        self._feat_drag_end = None
        self._drag_start_image = None
        self._last_image = None
        self.inception = inception_v3(
            pretrained=True,
            transform_input=False
        ).eval().to(self._device)
        for p in self.inception.parameters():
            p.requires_grad = False
        self.loss = []
        

    def render(self, **args):
        if self._disable_timing:
            self._is_timing = False
        else:
            self._start_event.record(torch.cuda.current_stream(self._device))
            self._is_timing = True
        res = dnnlib.EasyDict()
        try:
            init_net = False
            if not hasattr(self, 'G'):
                init_net = True
            if hasattr(self, 'pkl'):
                if self.pkl != args['pkl']:
                    init_net = True
            if hasattr(self, 'w_load'):
                if self.w_load is not args['w_load']:
                    init_net = True
            if hasattr(self, 'w0_seed'):
                if self.w0_seed != args['w0_seed']:
                    init_net = True
            if hasattr(self, 'w_plus'):
                if self.w_plus != args['w_plus']:
                    init_net = True
            if args['reset_w']:
                init_net = True
            res.init_net = init_net
            if init_net:
                self.init_network(res, **args)
            self._render_drag_impl(res, **args)
        except:
            res.error = CapturedException()
        if not self._disable_timing:
            self._end_event.record(torch.cuda.current_stream(self._device))
        if 'image' in res:
            res.image = self.to_cpu(res.image).detach().numpy()
            res.image = add_watermark_np(res.image, 'AI Generated')
        if 'stats' in res:
            res.stats = self.to_cpu(res.stats).detach().numpy()
        if 'error' in res:
            res.error = str(res.error)
        # if 'stop' in res and res.stop:

        if self._is_timing and not self._disable_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    def get_network(self, pkl, key, **tweak_kwargs):
        data = self._pkl_data.get(pkl, None)
        if data is None:
            print(f'Loading "{pkl}"... ', end='', flush=True)
            try:
                with dnnlib.util.open_url(pkl, verbose=False) as f:
                    data = legacy.load_network_pkl(f)
                print('Done.')
            except:
                data = CapturedException()
                print('Failed!')
            self._pkl_data[pkl] = data
            self._ignore_timing()
        if isinstance(data, CapturedException):
            raise data

        orig_net = data[key]
        cache_key = (orig_net, self._device, tuple(sorted(tweak_kwargs.items())))
        net = self._networks.get(cache_key, None)
        if net is None:
            try:
                if 'stylegan2' in pkl:
                    from training.networks_stylegan2 import Generator
                elif 'stylegan3' in pkl:
                    from training.networks_stylegan3 import Generator
                elif 'stylegan_human' in pkl:
                    from stylegan_human.training_scripts.sg2.training.networks import Generator
                else:
                    raise NameError('Cannot infer model type from pkl name!')

                print(data[key].init_args)
                print(data[key].init_kwargs)
                if 'stylegan_human' in pkl:
                    net = Generator(*data[key].init_args, **data[key].init_kwargs, square=False, padding=True)
                else:
                    net = Generator(*data[key].init_args, **data[key].init_kwargs)
                net.load_state_dict(data[key].state_dict())
                net.to(self._device)
            except:
                net = CapturedException()
            self._networks[cache_key] = net
            self._ignore_timing()
        if isinstance(net, CapturedException):
            raise net
        return net

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    def _ignore_timing(self):
        self._is_timing = False

    def _apply_cmap(self, x, name='viridis'):
        cmap = self._cmaps.get(name, None)
        if cmap is None:
            cmap = matplotlib.cm.get_cmap(name)
            cmap = cmap(np.linspace(0, 1, num=1024), bytes=True)[:, :3]
            cmap = self.to_device(torch.from_numpy(cmap))
            self._cmaps[name] = cmap
        hi = cmap.shape[0] - 1
        x = (x * hi + 0.5).clamp(0, hi).to(torch.int64)
        x = torch.nn.functional.embedding(x, cmap)
        return x

    def init_network(self, res,
        pkl             = None,
        w0_seed         = 0,
        w_load          = None,
        w_plus          = True,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        input_transform = None,
        lr              = 0.001,
        **kwargs
        ):
        # Dig up network details.
        self.pkl = pkl
        G = self.get_network(pkl, 'G_ema')
        self.G = G
        res.img_resolution = G.img_resolution
        res.num_ws = G.num_ws
        res.has_noise = any('noise_const' in name for name, _buf in G.synthesis.named_buffers())
        res.has_input_transform = (hasattr(G.synthesis, 'input') and hasattr(G.synthesis.input, 'transform'))

        # Set input transform.
        if res.has_input_transform:
            m = np.eye(3)
            try:
                if input_transform is not None:
                    m = np.linalg.inv(np.asarray(input_transform))
            except np.linalg.LinAlgError:
                res.error = CapturedException()
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # Generate random latents.
        self.w0_seed = w0_seed
        self.w_load = w_load

        if self.w_load is None:
            # Generate random latents.
            z = torch.from_numpy(np.random.RandomState(w0_seed).randn(1, 512)).to(self._device, dtype=self._dtype)

            # Run mapping network.
            label = torch.zeros([1, G.c_dim], device=self._device)
            w = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
        else:
            w = self.w_load.clone().to(self._device)

        self.w0 = w.detach().clone()
        self.w_plus = w_plus
        if w_plus:
            self.w = w.detach()
        else:
            self.w = w[:, 0, :].detach()
        self.w.requires_grad = True
        self.w_optim = torch.optim.Adam([self.w], lr=lr)

        self.feat_refs = None
        self.points0_pt = None

    def update_lr(self, lr):

        del self.w_optim
        self.w_optim = torch.optim.Adam([self.w], lr=lr)
        print(f'Rebuild optimizer with lr: {lr}')
        print('    Remain feat_refs and points0_pt')

    def _render_drag_impl(self, res,
        points          = [],       # 当前控制点位置
        targets         = [],       # 目标点位置
        mask            = None,     # 掩码，0 为保护
        lambda_mask     = 10,       # 掩码损失的权重
        reg             = 0,
        feature_idx     = 5,        # 默认第五层特征
        r1              = 3,        # 控制点影响范围
        r2              = 12,       # 控制点影响范围
        random_seed     = 0,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        force_fp32      = False,
        layer_name      = None,
        sel_channels    = 3,
        base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        untransform     = False,
        is_drag         = False,
        reset           = False,
        to_pil          = False,
        **kwargs
    ):
        G = self.G
        ws = self.w
        if ws.dim() == 2:
            ws = ws.unsqueeze(1).repeat(1,6,1)
        ws = torch.cat([ws[:,:6,:], self.w0[:,6:,:]], dim=1)
        if hasattr(self, 'points'):
            if len(points) != len(self.points):
                reset = True
        if reset:
            self.feat_refs = None
            self.points0_pt = None
            self._feat_drag_start = None
            self._drag_start_image = None
            self.loss = []
        self.points = points

        # Run synthesis network.
        label = torch.zeros([1, G.c_dim], device=self._device)
        img, feat = G(ws, label, truncation_psi=trunc_psi, noise_mode=noise_mode, input_is_w=True, return_feature=True)
        self._last_image = img.detach()

        h, w = G.img_resolution, G.img_resolution

        if is_drag and (reset or self._feat_drag_start is None):
            self._feat_drag_start = F.interpolate(
                feat[feature_idx].detach(),
                [h, w],
                mode='bilinear'
            )
        if is_drag and self._drag_start_image is None:
            self._drag_start_image = img.detach()

        if is_drag:
            X = torch.linspace(0, h, h)
            Y = torch.linspace(0, w, w)
            xx, yy = torch.meshgrid(X, Y)
            feat_resize = F.interpolate(feat[feature_idx], [h, w], mode='bilinear')
            # 每次调用时，覆盖前一次
            self._feat_drag_end = F.interpolate(
                feat[feature_idx].detach(),
                [h, w],
                mode='bilinear'
            )
            if self.feat_refs is None:
                self.feat0_resize = F.interpolate(feat[feature_idx].detach(), [h, w], mode='bilinear')
                self.feat_refs = []
                for point in points:
                    py, px = round(point[0]), round(point[1])
                    self.feat_refs.append(self.feat0_resize[:,:,py,px])
                self.points0_pt = torch.Tensor(points).unsqueeze(0).to(self._device) # 1, N, 2

            # Point tracking with feature matching
            if hasattr(self, 'raft_tracker') and self.raft_tracker is not None:
                try:
                    raft_points = self.raft_tracker.track(self.feat0_resize, feat_resize, points)
                except Exception as e:
                    print(f"[RAFT] 跟踪失败，错误: {e}， 回退到原始方法")
                    raft_points = points
            else:
                print("[RAFT] 跟踪器不可用，使用原始方法")
                raft_points = points
            # raft_points = points  # 加这一句，回退到原始方法
            with torch.no_grad():
                for j, point in enumerate(raft_points):
                    py = int(round(point[0]))
                    px = int(round(point[1]))
                    py = max(0, min(py, h - 1))
                    px = max(0, min(px, w - 1))
                    r = round(r2 / 512 * h)
                    up = max(py - r, 0)
                    down = min(py + r + 1, h)
                    left = max(px - r, 0)
                    right = min(px + r + 1, w)
                    feat_patch = feat_resize[:,:,up:down,left:right]
                    L2 = torch.linalg.norm(feat_patch - self.feat_refs[j].reshape(1,-1,1,1), dim=1)
                    _, idx = torch.min(L2.view(1,-1), -1)  # idx 展成了一维
                    width = right - left
                    point = [idx.item() // width + up, idx.item() % width + left]  # 需要转回来
                    points[j] = point

            res.points = [[point[0], point[1]] for point in points]

            # Motion supervision
            loss_motion = 0
            res.stop = True
            for j, point in enumerate(points):
                direction = torch.Tensor([targets[j][1] - point[1], targets[j][0] - point[0]])
                if torch.linalg.norm(direction) > max(2 / 512 * h, 2):
                    res.stop = False
                if torch.linalg.norm(direction) > 1:
                    distance = ((xx.to(self._device) - point[0])**2 + (yy.to(self._device) - point[1])**2)**0.5
                    relis, reljs = torch.where(distance < round(r1 / 512 * h))
                    direction = direction / (torch.linalg.norm(direction) + 1e-7)
                    gridh = (relis+direction[1]) / (h-1) * 2 - 1
                    gridw = (reljs+direction[0]) / (w-1) * 2 - 1
                    grid = torch.stack([gridw,gridh], dim=-1).unsqueeze(0).unsqueeze(0)
                    target = F.grid_sample(feat_resize.float(), grid, align_corners=True).squeeze(2)
                    loss_motion += F.l1_loss(feat_resize[:,:,relis,reljs].detach(), target)

            loss = loss_motion
            if mask is not None:
                if mask.min() == 0 and mask.max() == 1:
                    mask_usq = mask.to(self._device).unsqueeze(0).unsqueeze(0)
                    loss_fix = F.l1_loss(feat_resize * mask_usq, self.feat0_resize * mask_usq)
                    loss += lambda_mask * loss_fix

            loss += reg * F.l1_loss(ws, self.w0)  # latent code regularization
            self.loss.append(loss.item())
            if not res.stop:
                self.w_optim.zero_grad()
                loss.backward()
                self.w_optim.step()

        # Scale and convert to uint8.
        img = img[0]
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        if to_pil:
            from PIL import Image
            img = img.cpu().numpy()
            img = Image.fromarray(img)
        res.image = img
        res.w = ws.detach().cpu().numpy()

    def get_drag_features(self):
        return self._feat_drag_start, self._feat_drag_end

    @torch.no_grad()
    def evaluate_drag_quality(
        self,
        feat_before,     # Drag 前特征
        feat_after,      # Drag 后特征
        control_points,  # [(y, x), ...] 原 start 点
        target_points,   # [(y, x), ...] 目标点
    ):
        """
        返回每个点的 tracking error
        """
        assert self.raft_tracker is not None, "RAFT tracker not initialized"

        # 1. 用 RAFT 跟踪原控制点
        tracked_points = self.raft_tracker.track(
            feat_before,
            feat_after,
            control_points,
        )

        # 2. 计算与 target 的距离
        errors = []
        for (ty, tx), (y, x) in zip(target_points, tracked_points):
            dist = ((ty - y) ** 2 + (tx - x) ** 2) ** 0.5
            errors.append(dist)

        return {
            "tracked_points": tracked_points,
            "errors": errors,
            "mean_error": sum(errors) / len(errors) if errors else 0.0,
            "max_error": max(errors) if errors else 0.0,
        }

    def _extract_inception_feat(self, img):
        """
        img: torch.Tensor [1, 3, H, W], range [-1, 1]
        return: [1, 2048]
        """
        img = (img + 1) / 2
        img = F.interpolate(img, size=299, mode='bilinear', align_corners=False)
        feat = self.inception(img)
        return feat
    
    @torch.no_grad()
    def compute_pseudo_fid(self):
        assert self._drag_start_image is not None
        assert self._last_image is not None
        f0 = self._extract_inception_feat(self._drag_start_image)
        f1 = self._extract_inception_feat(self._last_image)
        return torch.norm(f0 - f1, p=2).item()

    def get_loss(self):
        return self.loss
    def save_experiment(
        self,
        exp_dir,
        losses,
        metrics,
        meta,
    ):
        os.makedirs(exp_dir, exist_ok=True)
        #from PIL import Image
        #tensor_to_pil(self._drag_start_image).save(
        #    os.path.join(exp_dir, "image_before.png")
        #)
        #tensor_to_pil(self._last_image).save(
        #    os.path.join(exp_dir, "image_after.png")
        #)
        np.save(os.path.join(exp_dir, "losses.npy"), np.array(losses))
        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(exp_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

#----------------------------------------------------------------------------
