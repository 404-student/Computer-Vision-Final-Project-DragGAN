Ubantu

sudo apt update, sudo apt upgrade -y

install wget gnupg software-properties-common

获取官方 CUDA 仓库，安装 12.1 的 cuda-Toolkit，因为官方给定 11.1 已经过时，会导致 compute_89 编译失败，bias_act 报错

安装 miniconda，创建并激活新的 conda 环境，安装 12.1 对应 PyTorch。

克隆 DragGAN 仓库，手动安装 yml 文件中的各种依赖，因为要避免安装错误的 PyTorch

按照 README，下载预训练模型

bias_act_plugin, upfirdn2d_plugin：CUDA 12.1 自带的 nvcc 未与系统默认 GCC 版本对齐。直接修改对应 py 文件：`__init__` 默认返回 False

安装旧版 Gradio：3.36.1

启动流程：

```
wsl
cd ~/DragGAN
conda activate stylegan3
nvidia-smi
python visualizer_drag_gradio.py
```

退出流程：

```
Ctrl + C
exit
```

获取原作者更新：`git fetch upstream, git merge upstream/main`

拉取我的仓库的最新代码：
```
git pull
git add .
git commit -m "wtf"
git push
```

latex 相关：在 [Texlive](https://tug.org/texlive/)，选 on DVD，在最下方 download，点第一个 CTAN mirror，选 202x.iso

开始探索！

当前进度：Code Understanding

visualizer_drag_gradio.py：

- 全局状态包括 images 图像，mask, last_mask 掩码，generator_params, params 核心：拖拽强度控制，控制点影响范围，空间类型（w, w+），交互状态：points, editing_state, pretrained_weight 当前选择模型，renderer 渲染引擎。
- 定义左侧区域：模型选择，生成控制，拖拽编辑，掩码编辑。暂时不是核心代码。
- UI 控件：别的都别管了，核心是 `on_click_start` 函数，重点阅读！其实也没有什么，核心函数在 renderer.py 的单步拖拽。

renderer.py：

- add_watermark_np：添加水印
- _is_timing: 居然在计时？这个以后可能有用
- init_network: 生成潜码

_render_drag_impl：重中之重！点跟踪：320，运动监督：348

整体流程：

- 首先加载模型与图像：先生成随机噪声，映射网络得到潜码（18 * 512），最后生成图像（512 * 512）。此时系统包括初始潜码 $w_0$，初始图像 $img_0$，初始特征图 $F_0$。什么是初始特征图？
- 提取第五层特征：形状为 [1, C, H_f, W_f]，其中 C 是通道数，H_f, W_f 是特征图分辨率。为什么？
- 点跟踪：首先获取点的特征向量。将特征图插值到图像分辨率（意思就是放大呗），放大成 [1, C, 512, 512]，然后取出对应点的值 [1, C, y, x]，也就是 [1, C] 的向量。然后就是在它的邻域（自定义半径 `r2`）找特征向量最相近的点，当然要用新的特征图去计算。
- 运动监督：先计算运动方向，这个简单，当前位置指向目标位置。怎么更新特征呢，假如要将 T 移动至 P，先计算所有点到 P 的欧式距离并找到内圈（自定义半径 `r1`）的像素。对这些像素，向运动方向移动一单位格，从当前特征图中采样

成功接入 RAFT！但效果怎么比较？

统计数据：

Mean tracking error: 38.118237105091836
Mean tracking error: 26.832815729997478

Mean tracking error: 46.010868281309364
Mean tracking error: 43.86732111857041

[217, 392] [149, 411]

Mean tracking error: 70.31687856711352
Mean tracking error: 79.58188507567081