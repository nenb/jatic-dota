from pathlib import Path
import os, shutil, torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

ROOT   = Path(__file__).parent.resolve()
SRC    = ROOT / "src"
VENDOR = SRC / "jatic_dota" / "_vendor" / "detectron2"
D2PKG  = VENDOR / "detectron2"
CSRC   = D2PKG / "layers" / "csrc"

PKG = "jatic_dota._vendor.detectron2.detectron2"

def rel(p: Path) -> str:
    return os.path.relpath(str(p), str(ROOT)).replace(os.sep, "/")

def copy_model_zoo_configs(vendor_root: Path = VENDOR, d2pkg: Path = D2PKG) -> None:
    """
    Upstream symlinks/copies top-level `configs/` into detectron2/model_zoo/configs.
    We copy during build so the files are present for package_data collection.
    """
    src = vendor_root / "configs"
    dst = d2pkg / "model_zoo" / "configs"
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)

copy_model_zoo_configs()

try:
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME
except Exception:
    CUDA_HOME = ROCM_HOME = None

def get_extensions():
    cpp_sources = [rel(p) for p in CSRC.rglob("*.cpp")]
    cu_sources  = [rel(p) for p in CSRC.rglob("*.cu")]

    is_rocm = (getattr(torch.version, "hip", None) is not None) and (ROCM_HOME is not None)
    use_cuda = (
        os.getenv("FORCE_CUDA") == "1" and os.uname().sysname != "Darwin"
    ) or (torch.cuda.is_available() and (CUDA_HOME is not None or is_rocm))

    ext_cls = CUDAExtension if use_cuda else CppExtension
    macros = [("WITH_HIP", None)] if is_rocm else ([("WITH_CUDA", None)] if use_cuda else [])
    extra = {"cxx": ["-O3"]}
    if use_cuda:
        extra["nvcc"] = [
            "-O3",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        if os.getenv("NVCC_FLAGS"):
            extra["nvcc"] += os.getenv("NVCC_FLAGS").split()

    sources = cpp_sources + (cu_sources if use_cuda else [])
    assert sources and all(not os.path.isabs(s) for s in sources), f"No sources under {CSRC}"

    return [ext_cls(
        name=f"{PKG}._C",
        sources=sources,
        include_dirs=[rel(CSRC)],
        define_macros=macros,
        extra_compile_args=extra,
    )]

setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)