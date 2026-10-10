# Origins and licensing

Recipe integration is derived from the MIT-licensed phase-one TensorFold recipe
by Anton Starikov. The TensorFold engine and model checkpoints remain external
artifacts with their own licenses and model terms; this source recipe does not
redistribute their weights or modify the original serving image.

The root MIT license applies to recipe files unless a file has an explicit
SPDX/license declaration. Files marked Apache-2.0 and vendored dependencies
retain their own terms, copyright notices and full license texts under
`third_party/`; the root license does not replace those declarations.

SparkRing is Apache-2.0. Prepared RoCEnante originates with Luke (`lukealonso`)
and Local Inference Lab contributors, with SparkRing's peer-path, proxy-health,
per-HCA GID and prepared execution integration. B12X supplies the genuine
preparation/compiler framework. Exact source pins, modifications and retained
licenses/notices are required for the isolated transport dependency package.

NVIDIA NCCL retains its upstream license and notices. The 2.32.3 dual-PCI-domain
patch is pinned from SparkRing, following the switchless NCCL work credited in
that project. Benchmark clients retain their upstream licenses and source pins.

Historical phase-one results are external references. A candidate CPU build or
source test is not labeled as a GPU, inference or performance result.
