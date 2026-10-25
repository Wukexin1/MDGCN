# Third-party attribution and notices

This file records upstream notices for material used by the experiment programs. It does not select a license for the authors' new code or assign a license to third-party datasets.

## MVGCN

Source: [wyn430/MVGCN](https://github.com/wyn430/MVGCN).

The complete and uniform model assemblies incorporate the supplied MVGCN/HGCNN and baseline implementations, with the compatibility edits recorded in each `assembly_manifest.json`. The public MVGCN repository identifies the following license. The full upstream notice was retrieved from its [LICENSE file](https://github.com/wyn430/MVGCN/blob/master/LICENSE) on 25 October 2026 (Git blob `e620cb0e523c13c5d6a53d45e1b609e394df6d1b`). This repository snapshot identifies the notice, not a claim that the local assembled source is identical to the current upstream file.

```text
MIT License

Copyright (c) 2023 wyn430

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## DGCNN

Source: [WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn).

MVGCN's README acknowledges this upstream graph-convolution implementation. The assembled programs contain the corresponding DGCNN/EdgeConv baseline operations. The notice below was retrieved from the upstream [LICENSE file](https://github.com/WangYueFt/dgcnn/blob/master/LICENSE) on 25 October 2026 (Git blob `4540a4900ca43fefd1312133d19e1820ac8334fa`).

```text
MIT License

Copyright (c) Massachusetts Institute of Technology and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## PointNet++ PyTorch implementation

Source: [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).

The PointNet++ utilities in the complete and uniform model assemblies contain code corresponding to the upstream [models/pointnet2_utils.py](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py). On 25 October 2026, comparison of parsed Python syntax confirmed that `square_distance` and the locally renamed `index_points_pt2` have the same function bodies as the upstream functions after removing docstrings and normalizing the function name. The local farthest-point sampling, ball-query and set-abstraction code follows the same implementation structure with adaptations, including deterministic sampling initialization, `torch.where` masking and inlined grouping. The local adaptations are recorded in `assembly_manifest.json`. The inspected upstream utility file has SHA-256 `640891adff1c045bed111a4d461ce23b9ec815067cef447a9d04b3e64239f0dd`. This source comparison identifies these incorporated portions; the full historical transfer chain is not recorded.

The complete upstream notice below was retrieved from its [LICENSE file](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/LICENSE) on 25 October 2026.

```text
MIT License

Copyright (c) 2019 benny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Dataset attribution

The original synthetic and real data remain attributed to the MVGCN authors. Their original download folders and the definitions of the derived inputs are listed in [docs/DATA.md](docs/DATA.md). Permission to share this study's additional preprocessing outputs and fixed splits has been provided by the data provider. The software notices above are not used as evidence of a blanket data-redistribution license. Private permission correspondence is retained outside the release.

## Items to complete before public release

- Complete the per-component provenance review using the identified sources above and the assembly manifests. These records identify incorporated portions and local edits; they do not establish a complete historical transfer chain for every baseline component.
- Select and add the license for the authors' new implementation and release tools.
- Preserve the applicable upstream notices when the code is packaged or redistributed.

Installed dependencies, including NumPy, PyTorch, SciPy, scikit-learn and Matplotlib, retain their respective licenses. Dependency binaries are not bundled in this candidate.
