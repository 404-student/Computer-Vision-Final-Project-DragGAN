<p align="center">
  <h1 align="center">基于 RAFT 的 DragGAN 点跟踪优化</h1>
  <p align="center">
    <a href="https://github.com/404-student"><strong>2400013148 谢嘉诚</strong></a>
  </p>
</p>

## 代码仓库

这是我的[代码仓库](https://github.com/404-student/Computer-Vision-Final-Project-DragGAN)。

## 运行方法

下面介绍我的代码的运行方法。首先，请将我的仓库下载到本地，由于我在 Windows 系统中下载时遇到了棘手的问题，而在 Linux 系统上较为顺利，因此请使用 Linux 系统下载，并确保系统中存在 Miniconda 与 NVIDIA 驱动。

其次，请安装必需的 CUDA 组件，在终端中如下逐行输入：

```sh
sudo apt install -y wget gnupg software-properties-common
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update
sudo apt install -y cuda-nvcc-12-1
sudo apt install -y cuda-cudart-dev-12-1
sudo apt install -y cuda-libraries-dev-12-1
```

然后，创建一个虚拟环境，并逐一安装需要的依赖项：

```sh
conda create -n stylegan3 python=3.9 -y
conda activate stylegan3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy click pillow==9.4.0 scipy==1.11.1 requests tqdm ninja matplotlib imageio
pip install imgui==2.0.0 glfw==2.6.1 gradio==3.35.2 pyopengl imageio-ffmpeg pyspng-seunglab
pip install gradio==3.36.1 "pydantic<2" "fastapi<0.100"
```

最后在该虚拟环境下，运行网页图形化界面即可。

```sh
python visualizer_drag_gradio.py
```

如果出现任何某依赖项的缺失等，请联系我，很有可能是因为系统差距等问题，因为我在按照原代码库配置环境时也出现了大量问题，因此我不保证该环境在任何设备下生效。为了杜绝一切隐患，我提供本机虚拟环境中的所有库，附在本文档的最后，因此如果实在运行不了，可以强行将环境调整为与我的环境一致的情况。

## 实验方法

我首先提供了报告中的实验结果。`spt_experiments` 是汽车实验的结果，带有图片与各种数据；`experiments_yes` 是使用改进点跟踪的 20 次实验结果，`experiments_no` 是使用原最近邻搜索的实验结果，均不含图片只含数据。助教老师可以修改 `clean_experiments.py` 文件中的根目录，然后运行脚本，即可获取与报告一致的结果。

其次，如果助教老师需要手动进行实验，需要在源代码中操作。具体来说，需要先取消 `visualizer_drag_gradio.py` 中 167-187 行的注释以启用随机生成点，取消 706-740 行的注释以启用自动暂停与保存文件。其次，目前启用的是改进后的 RAFT + 最近邻搜索，如果需要修改，取消 `renderer.py` 中 396 行的注释以切换回原模型，修改 `visualizer_drag_gradio.py` 中 276 行的 `r2` 值以达到与我的实验一致的效果。同时，我的代码支持实验的扩展，可以修改 `visualizer_drag_gradio.py` 中 171-177 行函数的参数以改变控制点数目、梯度阈值、以及控制点到目标点的距离最大值；可以取消 `renderer.py` 中 525-531 行的注释以在保存文件时保存原图与生成图。

最后，我的实验代码的自动化还不是很智能，需要很多手动操作。一方面，需要手动控制模型类型，因此每次实验时需要控制三个地方：第一个是要启动 RAFT；第二个是控制最近邻搜索的范围，不使用 RAFT 时需要扩大到 12，使用时需要缩小到 6；第三个是调整控制点的数目以加强实验结果。另一方面，需要手动调整文件夹，每次测试的文件都固定产生在 `experiments` 文件夹中，而使用脚本测试时只能对一个文件夹的所有文件计算，因此需要在更换变量时更改文件夹名称以分离不同实验下的文件。并且由于某些图片的上下有黑色区域，需要排除它们，因此还需要先保证控制点没有落在这种区域上，再进行实验。

调整好以后，可以直接启动图形界面，应当会自动生成一定的控制点与目标点，并且切换模型、重置图像等操作都会重新随机生成点，不过重置点能够清除掉这些点。同时，原先不支持自动停止的代码会在控制点停止的五十步后自动停止。**自动停止**的情况下会导出本次实验的所有信息到 `experiments/时间` 文件夹中，可以打开文件夹看单次实验的结果。也可以执行

```sh
python clean_experiments.py
```

自动计算 `experiments` 或自行指定的文件夹中所有实验的结果，包括 MD 和 FID 的均值与标准差，还可执行

```sh
python draw_loss.py --exp_dir <目标文件夹> (可选：--out_name <xxx.png>)
```

自动绘制一次实验的损失曲线。

## 一定能满足运行条件的环境

```
# Name                       Version          Build            Channel
_libgcc_mutex                0.1              main
_openmp_mutex                5.1              1_gnu
aiofiles                     25.1.0           pypi_0           pypi
aiohappyeyeballs             2.6.1            pypi_0           pypi
aiohttp                      3.13.2           pypi_0           pypi
aiosignal                    1.4.0            pypi_0           pypi
altair                       6.0.0            pypi_0           pypi
annotated-doc                0.0.4            pypi_0           pypi
annotated-types              0.7.0            pypi_0           pypi
anyio                        4.12.0           pypi_0           pypi
async-timeout                5.0.1            pypi_0           pypi
attrs                        25.4.0           pypi_0           pypi
bzip2                        1.0.8            h5eee18b_6
ca-certificates              2025.12.2        h06a4308_0
certifi                      2025.11.12       pypi_0           pypi
charset-normalizer           3.4.4            pypi_0           pypi
click                        8.1.8            pypi_0           pypi
contourpy                    1.3.0            pypi_0           pypi
cycler                       0.12.1           pypi_0           pypi
exceptiongroup               1.3.1            pypi_0           pypi
expat                        2.7.3            h7354ed3_4
fastapi                      0.99.1           pypi_0           pypi
ffmpy                        1.0.0            pypi_0           pypi
filelock                     3.19.1           pypi_0           pypi
fonttools                    4.60.2           pypi_0           pypi
frozenlist                   1.8.0            pypi_0           pypi
fsspec                       2025.10.0        pypi_0           pypi
glfw                         2.6.1            pypi_0           pypi
gradio                       3.36.1           pypi_0           pypi
gradio-client                1.3.0            pypi_0           pypi
h11                          0.16.0           pypi_0           pypi
hf-xet                       1.2.0            pypi_0           pypi
httpcore                     1.0.9            pypi_0           pypi
httpx                        0.28.1           pypi_0           pypi
huggingface-hub              1.2.3            pypi_0           pypi
idna                         3.11             pypi_0           pypi
imageio                      2.37.2           pypi_0           pypi
imageio-ffmpeg               0.6.0            pypi_0           pypi
imgui                        2.0.0            pypi_0           pypi
importlib-resources          6.5.2            pypi_0           pypi
jinja2                       3.1.6            pypi_0           pypi
jsonschema                   4.25.1           pypi_0           pypi
jsonschema-specifications    2025.9.1         pypi_0           pypi
kiwisolver                   1.4.7            pypi_0           pypi
ld_impl_linux-64             2.44             h153f514_2
libexpat                     2.7.3            h7354ed3_4
libffi                       3.4.4            h6a678d5_1
libgcc                       15.2.0           h69a1729_7
libgcc-ng                    15.2.0           h166f726_7
libgomp                      15.2.0           h4751f2c_7
libnsl                       2.0.0            h5eee18b_0
libstdcxx                    15.2.0           h39759b7_7
libstdcxx-ng                 15.2.0           hc03a8fd_7
libuuid                      1.41.5           h5eee18b_0
libxcb                       1.17.0           h9b100fa_0
libzlib                      1.3.1            hb25bd0a_0
linkify-it-py                2.0.3            pypi_0           pypi
markdown-it-py               2.2.0            pypi_0           pypi
markupsafe                   2.1.5            pypi_0           pypi
matplotlib                   3.9.4            pypi_0           pypi
mdit-py-plugins              0.3.3            pypi_0           pypi
mdurl                        0.1.2            pypi_0           pypi
mpmath                       1.3.0            pypi_0           pypi
multidict                    6.7.0            pypi_0           pypi
narwhals                     2.14.0           pypi_0           pypi
ncurses                      6.5              h7934f7d_0
networkx                     3.2.1            pypi_0           pypi
ninja                        1.13.0           pypi_0           pypi
numpy                        1.26.4           pypi_0           pypi
nvidia-cublas-cu12           12.1.3.1         pypi_0           pypi
nvidia-cuda-cupti-cu12       12.1.105         pypi_0           pypi
nvidia-cuda-nvrtc-cu12       12.1.105         pypi_0           pypi
nvidia-cuda-runtime-cu12     12.1.105         pypi_0           pypi
nvidia-cudnn-cu12            9.1.0.70         pypi_0           pypi
nvidia-cufft-cu12            11.0.2.54        pypi_0           pypi
nvidia-curand-cu12           10.3.2.106       pypi_0           pypi
nvidia-cusolver-cu12         11.4.5.107       pypi_0           pypi
nvidia-cusparse-cu12         12.1.0.106       pypi_0           pypi
nvidia-nccl-cu12             2.21.5           pypi_0           pypi
nvidia-nvjitlink-cu12        12.9.86          pypi_0           pypi
nvidia-nvtx-cu12             12.1.105         pypi_0           pypi
openssl                      3.0.18           hd6dcaed_0
orjson                       3.11.5           pypi_0           pypi
packaging                    25.0             pypi_0           pypi
pandas                       2.3.3            pypi_0           pypi
pillow                       9.4.0            pypi_0           pypi
pip                          25.3             pyhc872135_0
propcache                    0.4.1            pypi_0           pypi
pthread-stubs                0.3              h0ce48e5_1
pydantic                     1.10.26          pypi_0           pypi
pydub                        0.25.1           pypi_0           pypi
pygments                     2.19.2           pypi_0           pypi
pyopengl                     3.1.10           pypi_0           pypi
pyparsing                    3.3.1            pypi_0           pypi
pyspng-seunglab              1.1.2            pypi_0           pypi
python                       3.9.25           h0dcde21_1
python-dateutil              2.9.0.post0      pypi_0           pypi
python-multipart             0.0.20           pypi_0           pypi
pytz                         2025.2           pypi_0           pypi
pyyaml                       6.0.3            pypi_0           pypi
readline                     8.3              hc2a1206_0
referencing                  0.36.2           pypi_0           pypi
requests                     2.32.5           pypi_0           pypi
rpds-py                      0.27.1           pypi_0           pypi
scipy                        1.11.1           pypi_0           pypi
semantic-version             2.10.0           pypi_0           pypi
setuptools                   80.9.0           py39h06a4308_0
shellingham                  1.5.4            pypi_0           pypi
six                          1.17.0           pypi_0           pypi
sqlite                       3.51.0           h2a70700_0
starlette                    0.27.0           pypi_0           pypi
sympy                        1.13.1           pypi_0           pypi
tk                           8.6.15           h54e0aa7_0
torch                        2.5.1+cu121      pypi_0           pypi
torchvision                  0.20.1+cu121     pypi_0           pypi
tqdm                         4.67.1           pypi_0           pypi
triton                       3.1.0            pypi_0           pypi
typer-slim                   0.21.0           pypi_0           pypi
typing-extensions            4.15.0           pypi_0           pypi
typing-inspection            0.4.2            pypi_0           pypi
tzdata                       2025.3           pypi_0           pypi
uc-micro-py                  1.0.3            pypi_0           pypi
urllib3                      2.6.2            pypi_0           pypi
uvicorn                      0.39.0           pypi_0           pypi
websockets                   12.0             pypi_0           pypi
wheel                        0.45.1           py39h06a4308_0
xorg-libx11                  1.8.12           h9b100fa_1
xorg-libxau                  1.0.12           h9b100fa_0
xorg-libxdmcp                1.1.5            h9b100fa_0
xorg-xorgproto               2024.1           h5eee18b_1
xz                           5.6.4            h5eee18b_1
yarl                         1.22.0           pypi_0           pypi
zipp                         3.23.0           pypi_0           pypi
zlib                         1.3.1            hb25bd0a_0
```

## Acknowledgement

This code is developed based on [StyleGAN3](https://github.com/NVlabs/stylegan3). Part of the code is borrowed from [StyleGAN-Human](https://github.com/stylegan-human/StyleGAN-Human).

(cheers to the community as well)
## License

The code related to the DragGAN algorithm is licensed under [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/).
However, most of this project are available under a separate license terms: all codes used or modified from [StyleGAN3](https://github.com/NVlabs/stylegan3) is under the [Nvidia Source Code License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt).

Any form of use and derivative of this code must preserve the watermarking functionality showing "AI Generated".

## BibTeX

```bibtex
@inproceedings{pan2023draggan,
    title={Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold},
    author={Pan, Xingang and Tewari, Ayush, and Leimk{\"u}hler, Thomas and Liu, Lingjie and Meka, Abhimitra and Theobalt, Christian},
    booktitle = {ACM SIGGRAPH 2023 Conference Proceedings},
    year={2023}
}
```
