#!/usr/bin/env python3
"""Managed setup contract for the local Hunyuan3D-Part extension."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runtime.config import HostFacts, evaluate_host_support, resolve_runtime_context  # noqa: E402
from generator import Hunyuan3DPartGenerator  # noqa: E402
from runtime.platform_support import detect_cuda_toolkit, managed_bin_dir, managed_pip_path, managed_python_path, subprocess_command  # noqa: E402
from runtime.p3_sam import (  # noqa: E402
    SONATA_CACHE_ENV_VAR,
    UPSTREAM_REQUIRED_PATHS,
    build_adapter_readiness,
    runtime_source_root,
    sonata_cache_root,
)


SETUP_SUMMARY_NAME = ".modly-setup-summary.json"
NATIVE_BUILD_ARTIFACTS_DIRNAME = ".modly/native-build-artifacts"
TORCH_SCATTER_VERSION = "2.1.2"
TORCH_CLUSTER_VERSION = "1.6.3"
WINDOWS_SPCONV_CU120_PACKAGE = "spconv-cu120==2.3.6"
RUNTIME_PACKAGES = [
    "numpy",
    "trimesh",
    "safetensors",
    "huggingface_hub",
    "addict",
    "easydict",
    "omegaconf",
    "packaging",
    "scipy",
    "scikit-image",
    "scikit-learn",
    "tqdm",
    "fpsample",
    "numba",
    "viser",
    "gradio",
    "timm",
    "diffusers",
    "einops",
    "pymeshlab",
]
VERIFY_IMPORTS = [
    "numpy",
    "trimesh",
    "safetensors",
    "huggingface_hub",
    "addict",
    "easydict",
    "omegaconf",
    "packaging",
    "scipy",
    "skimage",
    "sklearn",
    "tqdm",
    "fpsample",
    "numba",
    "viser",
    "gradio",
    "torch",
    "torchvision",
    "timm",
    "diffusers",
    "einops",
    "pymeshlab",
]
CUMM_REPO_URL = "https://github.com/FindDefinition/cumm.git"
CUMM_REPO_REF = "v0.7.11"
CUMM_HEADER_SNAPSHOT_ROOT = ROOT / "vendor" / "cumm-header-snapshots"
CUMM_DTYPE_HEADER_LEGACY_VERSION = "v0.7.11"
CUMM_DTYPE_HEADER_CANONICAL_VERSION = "v0.8.2"
CUMM_TENSORVIEW_SENTINEL = Path("tensorview") / "tensor.h"
CUMM_COMMON_RELATIVE_PATH = Path("cumm") / "common.py"
CUMM_SUBINT_HEADER_RELATIVE_PATH = Path("tensorview") / "gemm" / "dtypes" / "subint.h"
CUMM_HALF_HEADER_RELATIVE_PATH = Path("tensorview") / "gemm" / "dtypes" / "half.h"
CUMM_BFLOAT16_HEADER_RELATIVE_PATH = Path("tensorview") / "gemm" / "dtypes" / "bfloat16.h"
CUMM_TF32_HEADER_RELATIVE_PATH = Path("tensorview") / "gemm" / "dtypes" / "tf32.h"
CUMM_FLOAT8_HEADER_RELATIVE_PATH = Path("tensorview") / "gemm" / "dtypes" / "float8.h"
CUMM_HOST_MATH_MACRO_NAMES = (
    "signbit",
    "isnan",
    "isfinite",
    "isinf",
    "isnormal",
    "fpclassify",
    "sqrt",
    "copysign",
)
CUMM_HALF_INCLUDE_SECTION = """#ifdef __CUDACC__
#if (__CUDACC_VER_MAJOR__ >= 10)
#include <cuda_fp16.h>
#endif
#endif
///////////////////////////////////////////////////////////////////////////////////////////////////
"""
CUMM_CUDA_RESOLVER_PATCH_SENTINEL = 'for env_name in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"):'
CUMM_CUDA_RESOLVER_LINUX_FRAGMENT = '''        else:
            try:
                nvcc_path = subprocess.check_output(["which", "nvcc"
                                                    ]).decode("utf-8").strip()
                lib = Path(nvcc_path).parent.parent / "lib"
                include = Path(nvcc_path).parent.parent / "targets/x86_64-linux/include"
                if lib.exists() and include.exists():
                    if (lib / "libcudart.so").exists() and (include / "cuda.h").exists():
                        # should be nvidia conda package
                        _CACHED_CUDA_INCLUDE_LIB = ([include], lib)
                        return _CACHED_CUDA_INCLUDE_LIB
            except:
                pass 

            linux_cuda_root = Path("/usr/local/cuda")
            include = linux_cuda_root / f"include"
            lib64 = linux_cuda_root / f"lib64"
            assert linux_cuda_root.exists(), f"can't find cuda in {linux_cuda_root} install via cuda installer or conda first."'''
CUMM_CUDA_RESOLVER_PATCHED_LINUX_FRAGMENT = '''        else:
            cuda_roots = []
            for env_name in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"):
                env_value = os.getenv(env_name, None)
                if env_value:
                    env_root = Path(env_value)
                    if env_root not in cuda_roots:
                        cuda_roots.append(env_root)
            try:
                nvcc_path = subprocess.check_output(["which", "nvcc"
                                                    ]).decode("utf-8").strip()
                nvcc_root = Path(nvcc_path).parent.parent
                if nvcc_root not in cuda_roots:
                    cuda_roots.insert(0, nvcc_root)
            except:
                pass 

            for cuda_root in cuda_roots:
                include_candidates = [
                    cuda_root / "targets/sbsa-linux/include",
                    cuda_root / "targets/x86_64-linux/include",
                    cuda_root / "include",
                ]
                lib_candidates = [
                    cuda_root / "targets/sbsa-linux/lib",
                    cuda_root / "lib64",
                    cuda_root / "lib",
                ]
                for include in include_candidates:
                    if not (include / "cuda.h").exists():
                        continue
                    for lib64 in lib_candidates:
                        if (lib64 / "libcudart.so").exists():
                            _CACHED_CUDA_INCLUDE_LIB = ([include], lib64)
                            return _CACHED_CUDA_INCLUDE_LIB

            linux_cuda_root = Path("/usr/local/cuda")
            include = linux_cuda_root / f"include"
            lib64 = linux_cuda_root / f"lib64"
            assert linux_cuda_root.exists(), f"can't find cuda in {linux_cuda_root} install via cuda installer or conda first."'''
CUMM_SUBINT_PATCH_ORIGINAL = """template <int Bits, bool Signed = true> struct integer_subbyte {

  /// Number of bits
  static int const kBits = Bits;

  /// Whether type is signed
  static bool const kSigned = Signed;

  /// External type
  using T = typename std::conditional<kSigned, int, unsigned>::type;

  /// Storage type
  using Storage = uint8_t;

  /// Bitmask used to truncate from larger integers
  static Storage const kMask = Storage((1 << kBits) - 1);

  //
  // Data members
  //

  Storage storage;

  //
  // Methods
  //

  /// No operation
  TV_HOST_DEVICE_INLINE
  integer_subbyte() {}

  /// Conversion from integer type
  TV_HOST_DEVICE_INLINE
  integer_subbyte(int value)
      : storage(reinterpret_cast<Storage const &>(value) & kMask) {}

  TV_HOST_DEVICE_INLINE
  integer_subbyte(unsigned value)
      : storage(reinterpret_cast<Storage const &>(value) & kMask) {}

  TV_HOST_DEVICE_INLINE
  integer_subbyte(double value) {
    T tmp = static_cast<T>(value);
    storage = Storage(reinterpret_cast<unsigned const &>(tmp) & kMask);
  }

  ///
  TV_HOST_DEVICE_INLINE
  operator T() const {
    if (kSigned) {
      // Sign extend
      if (storage & Storage(1 << (kBits - 1))) {
        return T(storage) | ~T(kMask);
      }
    }
    return T(storage);
  }

  /// Equality
  TV_HOST_DEVICE_INLINE
  bool operator==(integer_subbyte const &rhs) const {
    return storage == rhs.storage;
  }

  /// Inequality
  TV_HOST_DEVICE_INLINE
  bool operator!=(integer_subbyte const &rhs) const {
    return storage != rhs.storage;
  }

  /// Less than or equal
  TV_HOST_DEVICE_INLINE
  bool operator<=(integer_subbyte const &rhs) const {
    if (kSigned) {
      if (storage & (1 << (kBits - 1))) {
        return !(rhs.storage < storage);
      }
    }
    return storage < rhs.storage;
  }

  /// Less than
  TV_HOST_DEVICE_INLINE
  bool operator<(integer_subbyte const &rhs) const {
    if (kSigned) {
      if (storage & (1 << (kBits - 1))) {
        return !(rhs.storage <= storage);
      }
    }
    return storage < rhs.storage;
  }

  /// Greater than or equal
  TV_HOST_DEVICE_INLINE
  bool operator>=(integer_subbyte const &rhs) const { return !(*this < rhs); }

  /// Greater than
  TV_HOST_DEVICE_INLINE
  bool operator>(integer_subbyte const &rhs) const { return !(rhs < *this); }
};"""
CUMM_SUBINT_PATCH_REPLACEMENT_LEGACY = """template <int Bits, bool Signed = true> struct integer_subbyte {

  /// Number of bits
  static int const kBits = Bits;

  /// Whether type is signed
  static bool const kSigned = Signed;

  /// External type
  using self_type = integer_subbyte<Bits, Signed>;
  using T = typename std::conditional<Signed, int, unsigned>::type;

  /// Storage type
  using Storage = uint8_t;

  /// Bitmask used to truncate from larger integers
  static Storage const kMask = Storage((1 << kBits) - 1);

  //
  // Data members
  //

  Storage storage;

  //
  // Methods
  //

  /// No operation
  TV_HOST_DEVICE_INLINE
  integer_subbyte() : storage(0) {}

  /// Conversion from integer type
  TV_HOST_DEVICE_INLINE
  integer_subbyte(int value)
      : storage(static_cast<Storage>(static_cast<unsigned>(value) & unsigned(kMask))) {}

  TV_HOST_DEVICE_INLINE
  integer_subbyte(unsigned value)
      : storage(static_cast<Storage>(value & unsigned(kMask))) {}

  TV_HOST_DEVICE_INLINE
  integer_subbyte(double value) : integer_subbyte(static_cast<T>(value)) {}

  ///
  TV_HOST_DEVICE_INLINE
  T value() const {
    if (kSigned) {
      if (storage & Storage(1 << (kBits - 1))) {
        return T(storage) | ~T(kMask);
      }
    }
    return T(storage);
  }

  ///
  TV_HOST_DEVICE_INLINE
  explicit operator T() const { return value(); }

  /// Equality
  TV_HOST_DEVICE_INLINE
  bool operator==(self_type const &rhs) const { return storage == rhs.storage; }

  /// Inequality
  TV_HOST_DEVICE_INLINE
  bool operator!=(self_type const &rhs) const { return storage != rhs.storage; }

  /// Less than or equal
  TV_HOST_DEVICE_INLINE
  bool operator<=(self_type const &rhs) const { return value() <= rhs.value(); }

  /// Less than
  TV_HOST_DEVICE_INLINE
  bool operator<(self_type const &rhs) const { return value() < rhs.value(); }

  /// Greater than or equal
  TV_HOST_DEVICE_INLINE
  bool operator>=(self_type const &rhs) const { return value() >= rhs.value(); }

  /// Greater than
  TV_HOST_DEVICE_INLINE
  bool operator>(self_type const &rhs) const { return value() > rhs.value(); }
};"""
CUMM_SUBINT_PATCH_REPLACEMENT = """template <int Bits, bool Signed = true> struct integer_subbyte {

  /// Number of bits
  static int const kBits = Bits;

  /// Whether type is signed
  static bool const kSigned = Signed;

  /// External type
  using T = typename std::conditional<kSigned, int, unsigned>::type;

  /// Storage type
  using Storage = uint8_t;

  /// Bitmask used to truncate from larger integers
  static Storage const kMask = Storage((1u << kBits) - 1u);

  //
  // Data members
  //

  Storage storage;

  //
  // Methods
  //

  /// No operation
  TV_HOST_DEVICE_INLINE
  integer_subbyte() : storage(0) {}

  /// Conversion from integer type
  TV_HOST_DEVICE_INLINE
  integer_subbyte(int value)
      : storage(static_cast<Storage>(static_cast<unsigned>(value) & unsigned(kMask))) {}

  TV_HOST_DEVICE_INLINE
  integer_subbyte(unsigned value)
      : storage(static_cast<Storage>(value & unsigned(kMask))) {}

  TV_HOST_DEVICE_INLINE
  integer_subbyte(double value)
      : storage(static_cast<Storage>(static_cast<unsigned>(static_cast<T>(value)) & unsigned(kMask))) {}

  ///
  TV_HOST_DEVICE_INLINE
  operator T() const {
    if (kSigned && (storage & Storage(1u << (kBits - 1)))) {
      return T(storage) | ~T(kMask);
    }
    return T(storage);
  }

};"""
CUMM_SUBINT_BODY_START_MARKER = "/// 4-bit signed integer type\n"
CUMM_SUBINT_BODY_SUFFIX_MARKER = "\n///////////////////////////////////////////////////////////////////////////////////////////////////\n\n} // namespace tv"
CUMM_SUBINT_PATCH_BODY_ORIGINAL = f"""{CUMM_SUBINT_BODY_START_MARKER}{CUMM_SUBINT_PATCH_ORIGINAL}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// 1-bit Unsigned integer type
using uint1b_t = integer_subbyte<1, false>;

/// 2-bit Integer type
using int2b_t = integer_subbyte<2, true>;

/// 2-bit Unsigned integer type
using uint2b_t = integer_subbyte<2, false>;

/// 4-bit Integer type
using int4b_t = integer_subbyte<4, true>;

/// 4-bit Unsigned integer type
using uint4b_t = integer_subbyte<4, false>;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the size of an element in bits - specialized for uint1b_t
template <> struct sizeof_bits<uint1b_t> {{ static int const value = 1; }};

/// Defines the size of an element in bits - specialized for int2b_t
template <> struct sizeof_bits<int2b_t> {{ static int const value = 2; }};

/// Defines the size of an element in bits - specialized for uint2b_t
template <> struct sizeof_bits<uint2b_t> {{ static int const value = 2; }};

/// Defines the size of an element in bits - specialized for int4b_t
template <> struct sizeof_bits<int4b_t> {{ static int const value = 4; }};

/// Defines the size of an element in bits - specialized for uint4b_t
template <> struct sizeof_bits<uint4b_t> {{ static int const value = 4; }};"""
CUMM_SUBINT_PATCH_BODY_REPLACEMENT_LEGACY = f"""{CUMM_SUBINT_BODY_START_MARKER}{CUMM_SUBINT_PATCH_REPLACEMENT_LEGACY}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// 1-bit Unsigned integer type
using uint1b_t = integer_subbyte<1, false>;

/// 2-bit Integer type
using int2b_t = integer_subbyte<2, true>;

/// 2-bit Unsigned integer type
using uint2b_t = integer_subbyte<2, false>;

/// 4-bit Integer type
using int4b_t = integer_subbyte<4, true>;

/// 4-bit Unsigned integer type
using uint4b_t = integer_subbyte<4, false>;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the size of an element in bits - specialized for uint1b_t
template <> struct sizeof_bits<uint1b_t> {{ static int const value = 1; }};

/// Defines the size of an element in bits - specialized for int2b_t
template <> struct sizeof_bits<int2b_t> {{ static int const value = 2; }};

/// Defines the size of an element in bits - specialized for uint2b_t
template <> struct sizeof_bits<uint2b_t> {{ static int const value = 2; }};

/// Defines the size of an element in bits - specialized for int4b_t
template <> struct sizeof_bits<int4b_t> {{ static int const value = 4; }};

/// Defines the size of an element in bits - specialized for uint4b_t
template <> struct sizeof_bits<uint4b_t> {{ static int const value = 4; }};"""
CUMM_SUBINT_PATCH_BODY_REPLACEMENT = f"""{CUMM_SUBINT_BODY_START_MARKER}{CUMM_SUBINT_PATCH_REPLACEMENT}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// 1-bit Unsigned integer type
using uint1b_t = integer_subbyte<1, false>;

/// 2-bit Integer type
using int2b_t = integer_subbyte<2, true>;

/// 2-bit Unsigned integer type
using uint2b_t = integer_subbyte<2, false>;

/// 4-bit Integer type
using int4b_t = integer_subbyte<4, true>;

/// 4-bit Unsigned integer type
using uint4b_t = integer_subbyte<4, false>;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines the size of an element in bits - specialized for uint1b_t
template <> struct sizeof_bits<uint1b_t> {{ static int const value = 1; }};

/// Defines the size of an element in bits - specialized for int2b_t
template <> struct sizeof_bits<int2b_t> {{ static int const value = 2; }};

/// Defines the size of an element in bits - specialized for uint2b_t
template <> struct sizeof_bits<uint2b_t> {{ static int const value = 2; }};

/// Defines the size of an element in bits - specialized for int4b_t
template <> struct sizeof_bits<int4b_t> {{ static int const value = 4; }};

/// Defines the size of an element in bits - specialized for uint4b_t
template <> struct sizeof_bits<uint4b_t> {{ static int const value = 4; }};"""
GUARDED_FRAGMENT_SPEC = tuple[str, tuple[str, ...]]

CUMM_TF32_HEADER_CANONICAL_FRAGMENTS: tuple[GUARDED_FRAGMENT_SPEC, ...] = (
    (
        "#include <limits>\n#define CUDA_NAMESPACE_STD std\n",
        ("#include <limits>\n",),
    ),
    (
        "#include <tensorview/core/nvrtc_std.h>\n#define CUDA_NAMESPACE_STD cuda::std\n",
        ("#include <tensorview/core/nvrtc_std.h>\n",),
    ),
    (
        "namespace CUDA_NAMESPACE_STD {\n",
        ("namespace std {\n",),
    ),
)
CUMM_FLOAT8_HEADER_CANONICAL_FRAGMENTS: tuple[GUARDED_FRAGMENT_SPEC, ...] = (
    (
        "#include  <cstring>\n#define CUDA_NAMESPACE_STD std\n",
        ("#include  <cstring>\n",),
    ),
    (
        "#include <tensorview/core/nvrtc_std.h>\n#define CUDA_NAMESPACE_STD cuda::std\n",
        ("#include <tensorview/core/nvrtc_std.h>\n",),
    ),
    (
        "    static constexpr int FP8_NUM_EXPONENT_BITS = (T == FloatEncoding::E4M3) ? 4 : 5;\n",
        ("    static constexpr int FP8_NUM_EXPONENT_BITS = IS_E4M3 ? 4 : 5;\n",),
    ),
    (
        "    static constexpr int FP8_NUM_MANTISSA_BITS = (T == FloatEncoding::E4M3) ? 3 : 2;\n",
        ("    static constexpr int FP8_NUM_MANTISSA_BITS = IS_E4M3 ? 3 : 2;\n",),
    ),
    (
        "    static constexpr uint8_t  FP8_INFINITY_MASK = (T == FloatEncoding::E4M3) ? 0x78 : 0x7c;\n",
        ("    static constexpr uint8_t  FP8_INFINITY_MASK = IS_E4M3 ? 0x78 : 0x7c;\n",),
    ),
    (
        "    static constexpr int FP8_MAX_EXPONENT  = (T == FloatEncoding::E4M3) ?  7 :  15;\n",
        ("    static constexpr int FP8_MAX_EXPONENT  = IS_E4M3 ?  7 :  15;\n",),
    ),
    (
        "    static constexpr int FP8_MIN_EXPONENT  = (T == FloatEncoding::E4M3) ? -6 : -14;\n",
        ("    static constexpr int FP8_MIN_EXPONENT  = IS_E4M3 ? -6 : -14;\n",),
    ),
    (
        "    static constexpr int FP8_EXPONENT_BIAS = (T == FloatEncoding::E4M3) ?  7 :  15;\n",
        ("    static constexpr int FP8_EXPONENT_BIAS = IS_E4M3 ?  7 :  15;\n",),
    ),
    (
        "    static constexpr uint8_t FP8_MAX_FLT = ((T == FloatEncoding::E4M3) ? 0x7e : 0x7b);\n",
        (
            "    static constexpr uint8_t FP8_MAX_FLT = (IS_E4M3 ? 0x7e : 0x7b);\n",
            "    static constexpr uint8_t FP8_MAX_FLT = ((IS_E4M3) ? 0x7e : 0x7b);\n",
        ),
    ),
    (
        "        if (float8_base::isnan(flt)) {\n",
        ("        if (isnan(flt)) {\n",),
    ),
    (
        "        if (float8_base::isinf(flt)) {\n",
        ("        if (isinf(flt)) {\n",),
    ),
    (
        "namespace CUDA_NAMESPACE_STD {\n",
        ("namespace std {\n",),
    ),
)
ARM64_CU124_WHEELS = {
    "cp39": {
        "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp39-cp39-linux_aarch64.whl#sha256=012887a6190e562cb266d2210052c5deb5113f520a46dc2beaa57d76144a0e9b",
        "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp39-cp39-linux_aarch64.whl#sha256=e25b4ac3c9eec3f789f1c5491331dfe236b5f06a1f406ea82fa59fed4fc6f71e",
    },
    "cp310": {
        "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp310-cp310-linux_aarch64.whl#sha256=d468d0eddc188aa3c1e417ec24ce615c48c0c3f592b0354d9d3b99837ef5faa6",
        "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp310-cp310-linux_aarch64.whl#sha256=38765e53653f93e529e329755992ddbea81091aacedb61ed053f6a14efb289e5",
    },
    "cp311": {
        "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp311-cp311-linux_aarch64.whl#sha256=e080353c245b752cd84122e4656261eee6d4323a37cfb7d13e0fffd847bae1a3",
        "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp311-cp311-linux_aarch64.whl#sha256=2c5350a08abe005a16c316ae961207a409d0e35df86240db5f77ec41345c82f3",
    },
    "cp312": {
        "torch": "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp312-cp312-linux_aarch64.whl#sha256=302041d457ee169fd925b53da283c13365c6de75c6bb3e84130774b10e2fbb39",
        "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp312-cp312-linux_aarch64.whl#sha256=3e3289e53d0cb5d1b7f55b3f5912f46a08293c6791585ba2fc32c12cded9f9af",
    },
}
ARM64_CU128_WHEELS = {
    "cp39": {
        "torch": "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp39-cp39-manylinux_2_28_aarch64.whl#sha256=2f155388b1200e08f3e901bb3487ff93ca6d63cde87c29b97bb6762a8f63b373",
        "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp39-cp39-manylinux_2_28_aarch64.whl#sha256=7a398fad02f4ac6b7d18bea9a08dc14163ffc5a368618f29ceb0e53dfa91f69e",
    },
    "cp310": {
        "torch": "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp310-cp310-manylinux_2_28_aarch64.whl#sha256=b1f0cdd0720ad60536deb5baa427b782fd920dd4fcf72e244d32974caafa3b9e",
        "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp310-cp310-manylinux_2_28_aarch64.whl#sha256=566224d7b4f00bc6366bed1d62f834ca80f8e57fe41e10e4a5636bfa3ffb984e",
    },
    "cp311": {
        "torch": "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_aarch64.whl#sha256=47c895bcab508769d129d717a4b916b10225ae3855723aeec8dff8efe5346207",
        "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp311-cp311-manylinux_2_28_aarch64.whl#sha256=6be714bcdd8849549571f6acfaa2dfa9e00676f042bda517432745fb116f7904",
    },
    "cp312": {
        "torch": "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=6bba7dca5d9a729f1e8e9befb98055498e551efaf5ed034824c168b560afc1ac",
        "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=6e9752b48c1cdd7f6428bcd30c3d198b30ecea348d16afb651f95035e5252506",
    },
}
UPSTREAM_REPO_URL = "https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git"
UPSTREAM_REPO_REF = "main"
UPSTREAM_P3_SAM_MODEL_RELATIVE_PATH = Path("P3-SAM") / "model.py"
UPSTREAM_SONATA_MODEL_RELATIVE_PATH = Path("XPart") / "partgen" / "models" / "sonata" / "model.py"
UPSTREAM_MODEL_PATCH_OLD = "download_root='/root/sonata'"
UPSTREAM_MODEL_PATCH_NEW = (
    "download_root=os.environ.get("
    f"{SONATA_CACHE_ENV_VAR!r}, "
    "os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '.cache', 'sonata')))"
)
SONATA_FLASH_ATTENTION_LOAD_PATCH_OLD = "# ckpt[\"config\"]['enable_flash'] = False"
SONATA_FLASH_ATTENTION_LOAD_PATCH_NEW = 'ckpt["config"]["enable_flash"] = False'
SONATA_FLASH_ATTENTION_LOAD_MODEL_LINE = 'model = PointTransformerV3(**ckpt["config"])'
SONATA_FLASH_ATTENTION_LOAD_BY_CONFIG_PATCH_NEW = 'config["enable_flash"] = False'
SONATA_FLASH_ATTENTION_LOAD_BY_CONFIG_MODEL_LINE = 'model = PointTransformerV3(**config)'
SONATA_FLASH_ATTENTION_LOAD_BY_CONFIG_PREVIOUS_LINE = 'config = json.load(f)'

NATIVE_MODE_AUTO = "auto"
NATIVE_MODE_SKIP = "skip"
NATIVE_MODE_INSTALL = "install"
SUPPORTED_NATIVE_MODES = {NATIVE_MODE_AUTO, NATIVE_MODE_SKIP, NATIVE_MODE_INSTALL}
SUPPORTED = "supported"
UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class NativeHostLane:
    system: str
    machine: str
    platform_key: str
    python_tag: str
    gpu_sm: int
    cuda_version: int
    lane: str | None
    state: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeInstallStep:
    package: str
    requirement: str
    strategy: str
    status: str
    pip_args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pip_args"] = list(self.pip_args)
        payload["env"] = dict(self.env or {})
        return payload


@dataclass(frozen=True)
class NativeRuntimePlan:
    native_mode: str
    host: NativeHostLane
    desired_cuda_label: str | None
    toolkit_root: str | None
    nvcc_version: str | None
    toolkit_status: str
    toolkit_reason: str
    required_imports: tuple[str, ...]
    related_packages: tuple[str, ...]
    install_steps: tuple[NativeInstallStep, ...]
    install_strategy: str
    automatic_install_supported: bool
    status: str
    message: str
    next_action: str
    toolkit_include_dir: str | None = None
    toolkit_nvcc: str | None = None
    toolkit_lib_dirs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "native_mode": self.native_mode,
            "host": self.host.to_dict(),
            "desired_cuda_label": self.desired_cuda_label,
            "toolkit_root": self.toolkit_root,
            "toolkit_include_dir": self.toolkit_include_dir,
            "toolkit_nvcc": self.toolkit_nvcc,
            "toolkit_lib_dirs": list(self.toolkit_lib_dirs),
            "nvcc_version": self.nvcc_version,
            "toolkit_status": self.toolkit_status,
            "toolkit_reason": self.toolkit_reason,
            "required_imports": list(self.required_imports),
            "related_packages": list(self.related_packages),
            "install_steps": [step.to_dict() for step in self.install_steps],
            "install_strategy": self.install_strategy,
            "automatic_install_supported": self.automatic_install_supported,
            "status": self.status,
            "message": self.message,
            "next_action": self.next_action,
        }


@dataclass(frozen=True)
class NativeRuntimeProbe:
    ready: bool
    status: str
    managed_python: str
    imports: dict[str, dict[str, Any]]
    missing_modules: tuple[str, ...]
    blocked_imports: tuple[str, ...]
    message: str
    next_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "status": self.status,
            "managed_python": self.managed_python,
            "imports": self.imports,
            "missing_modules": list(self.missing_modules),
            "blocked_imports": list(self.blocked_imports),
            "message": self.message,
            "next_action": self.next_action,
        }


def venv_python(venv_dir: Path) -> Path:
    return managed_python_path(venv_dir, platform.system())


def venv_bin_dir(venv_dir: Path) -> Path:
    return managed_bin_dir(venv_dir, platform.system())


def venv_pip(venv_dir: Path) -> Path:
    return managed_pip_path(venv_dir, platform.system())


def pip_run(venv_dir: Path, *args: str) -> None:
    subprocess.run(subprocess_command(venv_python(venv_dir), "-m", "pip", *args), check=True)


def pip_install(venv_dir: Path, *packages: str, index_url: str | None = None, extra_index_url: str | None = None) -> None:
    command = ["install", "--no-cache-dir"]
    if index_url:
        command.extend(["--index-url", index_url])
    if extra_index_url:
        command.extend(["--extra-index-url", extra_index_url])
    command.extend(packages)
    pip_run(venv_dir, *command)


def _native_build_artifact_root(extension_dir: Path) -> Path:
    return extension_dir / NATIVE_BUILD_ARTIFACTS_DIRNAME


def _native_build_artifact_slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "artifact").strip())
    return normalized.strip(".-") or "artifact"


def _native_build_preservation(package: str, extension_dir: Path) -> dict[str, Any]:
    artifact_root = _native_build_artifact_root(extension_dir)
    package_slug = _native_build_artifact_slug(package)
    temp_dir = artifact_root / "tmp" / package_slug
    build_tracker_dir = artifact_root / "pip-build-tracker" / package_slug
    logs_dir = artifact_root / "logs"
    stdout_log_path = logs_dir / f"{package_slug}.stdout.log"
    stderr_log_path = logs_dir / f"{package_slug}.stderr.log"

    for path in (temp_dir, build_tracker_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "root": str(artifact_root),
        "temp_dir": str(temp_dir),
        "build_tracker_dir": str(build_tracker_dir),
        "stdout_log_path": str(stdout_log_path),
        "stderr_log_path": str(stderr_log_path),
        "env": {
            "TMPDIR": str(temp_dir),
            "TEMP": str(temp_dir),
            "TMP": str(temp_dir),
            "PIP_BUILD_TRACKER": str(build_tracker_dir),
            "PIP_NO_CLEAN": "1",
            "PIP_VERBOSE": "1",
            "CMAKE_VERBOSE_MAKEFILE": "ON",
            "VERBOSE": "1",
        },
    }


def _write_native_build_log(path: str | Path, content: str) -> None:
    Path(path).write_text(content or "", encoding="utf-8")


def summarize_native_build_artifacts(*, extension_dir: Path, native_runtime: dict[str, Any]) -> dict[str, Any]:
    install_results = native_runtime.get("install_results") or []
    plan = native_runtime.get("plan") or {}
    steps: list[dict[str, Any]] = []
    resolver_patch: dict[str, Any] | None = None
    for result in install_results:
        package = result.get("package")
        if package == "cumm-cuda-resolver-patch":
            resolver_patch = {
                "package": package,
                "status": result.get("status"),
                "path": result.get("path"),
                "action": result.get("action"),
                "selected_toolkit_root": result.get("selected_toolkit_root"),
                "selected_include_dir": result.get("selected_include_dir"),
                "selected_nvcc_path": result.get("selected_nvcc_path"),
                "no_global_cuda_mutation": bool(result.get("no_global_cuda_mutation", False)),
                "reason": result.get("reason"),
            }
        artifact_root = result.get("artifact_root")
        if not package or not artifact_root:
            continue
        steps.append(
            {
                "package": package,
                "status": result.get("status"),
                "command": result.get("command") or [],
                "cwd": result.get("cwd"),
                "artifact_root": artifact_root,
                "artifact_tmp_dir": result.get("artifact_tmp_dir"),
                "build_tracker_dir": result.get("build_tracker_dir"),
                "stdout_log_path": result.get("stdout_log_path"),
                "stderr_log_path": result.get("stderr_log_path"),
            }
        )
    return {
        "root": str(_native_build_artifact_root(extension_dir)),
        "cuda": {
            "desired_label": plan.get("desired_cuda_label"),
            "toolkit_root": plan.get("toolkit_root"),
            "include_dir": plan.get("toolkit_include_dir"),
            "nvcc_path": plan.get("toolkit_nvcc"),
            "nvcc_version": plan.get("nvcc_version"),
            "toolkit_status": plan.get("toolkit_status"),
            "toolkit_reason": plan.get("toolkit_reason"),
            "resolver_patch": resolver_patch,
        },
        "steps": steps,
    }


def python_tag(python_exe: str | Path) -> str:
    return subprocess.check_output(
        [str(python_exe), "-c", "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"],
        text=True,
    ).strip()


def _parse_payload(raw_value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _parse_compute_capability_to_sm(raw_value: str) -> int | None:
    match = re.search(r"(?P<major>\d+)\.(?P<minor>\d+)", str(raw_value).strip())
    if not match:
        return None
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    return (major * 10) + minor


def _parse_highest_compute_capability(raw_output: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for raw_line in (raw_output or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        gpu_sm = _parse_compute_capability_to_sm(line)
        if gpu_sm is None:
            continue
        candidates.append({"compute_capability": line, "gpu_sm": gpu_sm})

    selected = max(candidates, key=lambda item: item["gpu_sm"], default=None)
    return {
        "compute_capabilities": [item["compute_capability"] for item in candidates],
        "gpu_sm": int(selected["gpu_sm"]) if selected else 0,
        "selected_compute_capability": selected["compute_capability"] if selected else None,
        "detected": selected is not None,
    }


def _parse_cuda_version_output(raw_output: str) -> dict[str, Any]:
    match = re.search(r"CUDA Version\s*:\s*(?P<version>\d+\.\d+)", raw_output or "")
    if not match:
        return {"cuda_version": 0, "cuda_version_text": None, "detected": False}

    version_text = match.group("version")
    cuda_version = _parse_compute_capability_to_sm(version_text) or 0
    return {
        "cuda_version": cuda_version,
        "cuda_version_text": version_text,
        "detected": cuda_version > 0,
    }


def _run_nvidia_smi(*args: str) -> dict[str, Any]:
    command = ["nvidia-smi", *args]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "error": "nvidia-smi not found",
        }
    except Exception as exc:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "ok": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": None if result.returncode == 0 else (result.stderr.strip() or result.stdout.strip() or "nvidia-smi failed"),
    }


def detect_gpu_runtime() -> dict[str, Any]:
    compute_result = _run_nvidia_smi("--query-gpu=compute_cap", "--format=csv,noheader")
    compute_details = _parse_highest_compute_capability(compute_result.get("stdout", ""))

    cuda_result = _run_nvidia_smi("--version")
    cuda_details = _parse_cuda_version_output(cuda_result.get("stdout", ""))

    errors = [
        detail
        for detail in (compute_result.get("error"), cuda_result.get("error"))
        if detail
    ]
    detected = bool(compute_details["detected"] or cuda_details["detected"])
    return {
        "detected": detected,
        "gpu_sm": int(compute_details["gpu_sm"]),
        "cuda_version": int(cuda_details["cuda_version"]),
        "selected_compute_capability": compute_details["selected_compute_capability"],
        "compute_capabilities": compute_details["compute_capabilities"],
        "cuda_version_text": cuda_details["cuda_version_text"],
        "commands": {
            "compute_capability": compute_result,
            "cuda_version": cuda_result,
        },
        "errors": errors,
        "status": "detected" if detected else "unavailable",
    }


def _resolved_numeric_detail(*, field: str, explicit_value: int | None, detected_value: int, detection: dict[str, Any]) -> dict[str, Any]:
    explicit_valid = explicit_value not in (None, 0)
    resolved_value = int(explicit_value if explicit_valid else detected_value or 0)
    source = "payload" if explicit_valid else ("nvidia-smi" if detected_value else "default")
    detail = {
        "field": field,
        "value": resolved_value,
        "source": source,
        "payload_value": 0 if explicit_value is None else int(explicit_value),
        "detected_value": int(detected_value or 0),
    }
    if not explicit_valid and detection.get("errors"):
        detail["detection_errors"] = list(detection["errors"])
    return detail


def resolve_gpu_runtime_details(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    source = payload or {}
    explicit_gpu_sm = _parse_optional_int(source.get("gpu_sm"))
    explicit_cuda_version = _parse_optional_int(source.get("cuda_version"))
    needs_detection = explicit_gpu_sm in (None, 0) or explicit_cuda_version in (None, 0)
    detection = detect_gpu_runtime() if needs_detection else {
        "detected": False,
        "gpu_sm": 0,
        "cuda_version": 0,
        "selected_compute_capability": None,
        "compute_capabilities": [],
        "cuda_version_text": None,
        "commands": {},
        "errors": [],
        "status": "skipped",
    }

    gpu_sm_detail = _resolved_numeric_detail(
        field="gpu_sm",
        explicit_value=explicit_gpu_sm,
        detected_value=int(detection.get("gpu_sm") or 0),
        detection=detection,
    )
    cuda_version_detail = _resolved_numeric_detail(
        field="cuda_version",
        explicit_value=explicit_cuda_version,
        detected_value=int(detection.get("cuda_version") or 0),
        detection=detection,
    )

    return {
        "gpu_sm": gpu_sm_detail["value"],
        "cuda_version": cuda_version_detail["value"],
        "gpu_sm_detail": gpu_sm_detail,
        "cuda_version_detail": cuda_version_detail,
        "nvidia_smi": detection,
    }


def _platform_facts() -> tuple[str, str]:
    return platform.system(), normalize_machine(platform.machine())


def normalize_machine(raw: str) -> str:
    value = raw.strip().lower()
    if value in {"amd64", "x64"}:
        return "x86_64"
    if value == "aarch64":
        return "arm64"
    return value


def _is_linux_arm64(system_name: str, machine: str) -> bool:
    return system_name == "Linux" and normalize_machine(machine) == "arm64"


def _is_windows_amd64(system_name: str, machine: str) -> bool:
    return system_name == "Windows" and normalize_machine(machine) == "x86_64"


def _native_required_imports() -> tuple[str, ...]:
    return ("spconv.pytorch", "torch_scatter", "torch_cluster")


def _native_related_packages() -> tuple[str, ...]:
    return ("cumm",)


def _native_probe_modules() -> tuple[str, ...]:
    return ("spconv", *_native_required_imports(), *_native_related_packages())


def _gpu_sm_to_cuda_arch(gpu_sm: int) -> str:
    normalized = max(int(gpu_sm or 0), 0)
    major, minor = divmod(normalized, 10)
    return f"{major}.{minor}"


def _torch_cuda_arch_list_value(gpu_sm: int) -> str | None:
    cuda_arch = _gpu_sm_to_cuda_arch(gpu_sm)
    return None if cuda_arch == "12.1" else cuda_arch


def _cumm_supported_cuda_arches() -> tuple[str, ...]:
    return (
        "3.5",
        "3.7",
        "5.0",
        "5.2",
        "5.3",
        "6.0",
        "6.1",
        "6.2",
        "7.0",
        "7.2",
        "7.5",
        "8.0",
        "8.6",
        "8.7",
        "8.9",
        "9.0",
    )


def _cumm_cuda_arch_list_value(gpu_sm: int) -> str | None:
    cuda_arch = _gpu_sm_to_cuda_arch(gpu_sm)
    return cuda_arch if cuda_arch in _cumm_supported_cuda_arches() else None


def _cumm_arch_override_note(gpu_sm: int) -> str | None:
    cuda_arch = _gpu_sm_to_cuda_arch(gpu_sm)
    if _cumm_cuda_arch_list_value(gpu_sm) is not None:
        return None
    return (
        f"Omitted CUMM_CUDA_ARCH_LIST for gpu_sm={int(gpu_sm or 0)} ({cuda_arch}) because cumm/spconv "
        "v0.7.11/v2.3.8 only validate explicit cumm arches through 9.0; build will rely on upstream defaults."
    )


def _prepend_env_path(value: str, current: str | None = None) -> str:
    current_value = str(current or "").strip()
    return f"{value}{os.pathsep}{current_value}" if current_value else value


def _unique_path_entries(*groups: str | None) -> list[str]:
    entries: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for raw_entry in str(group or "").split(os.pathsep):
            entry = raw_entry.strip()
            if not entry or entry in seen:
                continue
            seen.add(entry)
            entries.append(entry)
    return entries


def _join_path_entries(*groups: str | None, blocked: set[str] | None = None) -> str:
    blocked_entries = {item.strip() for item in (blocked or set()) if item and item.strip()}
    return os.pathsep.join(entry for entry in _unique_path_entries(*groups) if entry not in blocked_entries)


def _prepend_flag(value: str, current: str | None = None) -> str:
    current_value = str(current or "").strip()
    return f"{value} {current_value}" if current_value else value


def _parse_nvcc_release(raw_output: str) -> str | None:
    match = re.search(r"release\s+(?P<version>\d+\.\d+)", raw_output or "")
    return match.group("version") if match else None


def _read_nvcc_release(nvcc_path: Path) -> str | None:
    try:
        result = subprocess.run([str(nvcc_path), "--version"], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return _parse_nvcc_release(result.stdout or result.stderr)


def _cuda_toolkit_lib_dirs(toolkit_root: Path) -> list[Path]:
    if platform.system() == "Windows":
        return [toolkit_root / "lib" / "x64"]
    return [
        toolkit_root / "lib64",
        toolkit_root / "targets" / "sbsa-linux" / "lib",
    ]


def _cuda_toolkit_include_dir(toolkit_root: Path) -> Path:
    return toolkit_root / "include"


def _cuda_toolkit_nvcc_path(toolkit_root: Path) -> Path:
    return toolkit_root / "bin" / ("nvcc.exe" if platform.system() == "Windows" else "nvcc")


def _cuda_toolkit_metadata(toolkit_root: str | None, lib_dirs: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    if not toolkit_root:
        return {
            "toolkit_root": None,
            "include_dir": None,
            "nvcc_path": None,
            "lib_dirs": [],
        }
    toolkit_path = Path(toolkit_root)
    return {
        "toolkit_root": toolkit_root,
        "include_dir": str(_cuda_toolkit_include_dir(toolkit_path)),
        "nvcc_path": str(_cuda_toolkit_nvcc_path(toolkit_path)),
        "lib_dirs": list(lib_dirs or []),
    }


def resolve_cuda_toolkit_for_label(label: str | None) -> dict[str, Any]:
    if not label:
        return {
            "desired_cuda_label": None,
            "toolkit_root": None,
            "include_dir": None,
            "nvcc_path": None,
            "nvcc_version": None,
            "status": "not-applicable",
            "reason": "No CUDA toolkit selection is needed for an unsupported native lane.",
            "lib_dirs": [],
        }

    if platform.system() == "Windows":
        detected = detect_cuda_toolkit("Windows", os.environ)
        if detected.get("ready"):
            nvcc_version = _read_nvcc_release(Path(str(detected["nvcc_path"])))
            return {
                "desired_cuda_label": label,
                "toolkit_root": detected.get("toolkit_root"),
                "include_dir": str(Path(str(detected["toolkit_root"])) / "include"),
                "nvcc_path": detected.get("nvcc_path"),
                "nvcc_version": nvcc_version,
                "status": "matched",
                "reason": f"Selected Windows CUDA toolkit from {detected.get('source')} for readiness/probe compatibility.",
                "lib_dirs": list(detected.get("lib_dirs") or []),
            }
        return {
            "desired_cuda_label": label,
            "toolkit_root": None,
            "include_dir": None,
            "nvcc_path": None,
            "nvcc_version": None,
            "status": "blocked",
            "reason": str(detected.get("reason")),
            "lib_dirs": [],
            "next_action": detected.get("next_action"),
        }

    candidates: list[tuple[Path, str, bool]] = []
    if label == "cu128":
        candidates = [(Path("/usr/local/cuda-12.8"), "12.8", False)]
    elif label == "cu124":
        candidates = [
            (Path("/usr/local/cuda-12.4"), "12.4", False),
            (Path("/usr/local/cuda"), "12.4", True),
        ]
    else:
        return {
            "desired_cuda_label": label,
            "toolkit_root": None,
            "include_dir": None,
            "nvcc_path": None,
            "nvcc_version": None,
            "status": "blocked",
            "reason": f"No CUDA toolkit resolver is encoded for PyTorch lane {label}.",
            "lib_dirs": [],
        }

    inspected: list[str] = []
    for toolkit_root, expected_release, allow_default_symlink in candidates:
        nvcc_path = toolkit_root / "bin" / "nvcc"
        if not nvcc_path.exists():
            inspected.append(f"{toolkit_root} missing bin/nvcc")
            continue
        release = _read_nvcc_release(nvcc_path)
        if release == expected_release:
            lib_dirs = [str(path) for path in _cuda_toolkit_lib_dirs(toolkit_root) if path.exists()]
            metadata = _cuda_toolkit_metadata(str(toolkit_root), lib_dirs)
            return {
                "desired_cuda_label": label,
                **metadata,
                "nvcc_version": release,
                "status": "matched",
                "reason": (
                    f"Selected {toolkit_root} because nvcc reports CUDA {release} for the {label} PyTorch lane."
                ),
            }
        symlink_note = " (default /usr/local/cuda fallback)" if allow_default_symlink else ""
        inspected.append(
            f"{toolkit_root} reports {release or 'unknown'} instead of required {expected_release}{symlink_note}"
        )

    return {
        "desired_cuda_label": label,
        "toolkit_root": None,
        "include_dir": None,
        "nvcc_path": None,
        "nvcc_version": None,
        "status": "blocked",
        "reason": (
            f"No CUDA toolkit matching PyTorch lane {label} was found. "
            f"Inspected: {'; '.join(inspected) if inspected else 'no candidates'}"
        ),
        "lib_dirs": [],
    }


def _native_common_build_env(*, toolkit: dict[str, Any], venv_dir: Path | None = None) -> dict[str, str]:
    build_env = {
        "FORCE_CUDA": "1",
        "MAX_JOBS": "2",
        "CMAKE_BUILD_PARALLEL_LEVEL": "2",
    }
    path_value = os.environ.get("PATH")
    toolkit_root = toolkit.get("toolkit_root")
    if toolkit_root:
        bin_dir = f"{toolkit_root}/bin"
        include_dir = str(toolkit.get("include_dir") or _cuda_toolkit_include_dir(Path(toolkit_root)))
        nvcc_path = str(toolkit.get("nvcc_path") or _cuda_toolkit_nvcc_path(Path(toolkit_root)))
        ld_entries = list(toolkit.get("lib_dirs") or [])
        blocked_include_paths = {"/usr/local/cuda/include"} if toolkit_root != "/usr/local/cuda" else set()
        blocked_library_paths = {"/usr/local/cuda/lib64", "/usr/local/cuda/targets/sbsa-linux/lib"} if toolkit_root != "/usr/local/cuda" else set()
        path_value = _prepend_env_path(bin_dir, path_value)
        if venv_dir is not None:
            path_value = _prepend_env_path(str(venv_bin_dir(venv_dir)), path_value)
        build_env.update(
            {
                "CUDA_HOME": toolkit_root,
                "CUDA_PATH": toolkit_root,
                "CUDA_ROOT": toolkit_root,
                "CUDA_BIN_PATH": toolkit_root,
                "CUDA_TOOLKIT_ROOT_DIR": toolkit_root,
                "CUDACXX": nvcc_path,
                "CMAKE_CUDA_COMPILER": nvcc_path,
                "CUDA_NVCC_EXECUTABLE": nvcc_path,
                "PATH": path_value,
                "CPATH": _join_path_entries(include_dir, os.environ.get("CPATH"), blocked=blocked_include_paths),
                "C_INCLUDE_PATH": _join_path_entries(include_dir, os.environ.get("C_INCLUDE_PATH"), blocked=blocked_include_paths),
                "CPLUS_INCLUDE_PATH": _join_path_entries(include_dir, os.environ.get("CPLUS_INCLUDE_PATH"), blocked=blocked_include_paths),
                "LIBRARY_PATH": _join_path_entries(os.pathsep.join(ld_entries), os.environ.get("LIBRARY_PATH"), blocked=blocked_library_paths),
                "LD_LIBRARY_PATH": _join_path_entries(os.pathsep.join(ld_entries), os.environ.get("LD_LIBRARY_PATH"), blocked=blocked_library_paths),
                "CPPFLAGS": _prepend_flag(f"-isystem {include_dir}", os.environ.get("CPPFLAGS")),
                "CFLAGS": _prepend_flag(f"-isystem {include_dir}", os.environ.get("CFLAGS")),
                "CXXFLAGS": _prepend_flag(f"-isystem {include_dir}", os.environ.get("CXXFLAGS")),
                "NVCCFLAGS": _prepend_flag(f"-I{include_dir}", os.environ.get("NVCCFLAGS")),
                "CUDAFLAGS": _prepend_flag(f"-I{include_dir}", os.environ.get("CUDAFLAGS")),
            }
        )
    elif venv_dir is not None:
        build_env["PATH"] = _prepend_env_path(str(venv_bin_dir(venv_dir)), path_value)
    elif path_value:
        build_env["PATH"] = path_value
    return build_env


def _torch_pyg_build_env(*, gpu_sm: int, toolkit: dict[str, Any], venv_dir: Path | None = None) -> dict[str, str]:
    build_env = _native_common_build_env(toolkit=toolkit, venv_dir=venv_dir)
    torch_cuda_arch = _torch_cuda_arch_list_value(gpu_sm)
    if torch_cuda_arch:
        build_env["TORCH_CUDA_ARCH_LIST"] = torch_cuda_arch
    return build_env


def _cumm_spconv_build_env(*, gpu_sm: int, toolkit: dict[str, Any], venv_dir: Path | None = None) -> dict[str, str]:
    build_env = _native_common_build_env(toolkit=toolkit, venv_dir=venv_dir)
    cumm_cuda_arch = _cumm_cuda_arch_list_value(gpu_sm)
    if cumm_cuda_arch:
        build_env["CUMM_CUDA_ARCH_LIST"] = cumm_cuda_arch
    return build_env


def _cumm_build_env(*, gpu_sm: int, toolkit: dict[str, Any], venv_dir: Path | None = None) -> dict[str, str]:
    build_env = _cumm_spconv_build_env(gpu_sm=gpu_sm, toolkit=toolkit, venv_dir=venv_dir)
    build_env["CUMM_DISABLE_JIT"] = "1"
    return build_env


def _spconv_build_env(*, gpu_sm: int, toolkit: dict[str, Any], venv_dir: Path | None = None) -> dict[str, str]:
    build_env = _cumm_spconv_build_env(gpu_sm=gpu_sm, toolkit=toolkit, venv_dir=venv_dir)
    build_env["SPCONV_DISABLE_JIT"] = "1"
    return build_env


def _linux_arm64_native_install_steps(*, gpu_sm: int, toolkit: dict[str, Any], venv_dir: Path | None = None) -> tuple[NativeInstallStep, ...]:
    common_env = _native_common_build_env(toolkit=toolkit, venv_dir=venv_dir)
    torch_pyg_env = _torch_pyg_build_env(gpu_sm=gpu_sm, toolkit=toolkit, venv_dir=venv_dir)
    cumm_env = _cumm_build_env(gpu_sm=gpu_sm, toolkit=toolkit, venv_dir=venv_dir)
    spconv_env = _spconv_build_env(gpu_sm=gpu_sm, toolkit=toolkit, venv_dir=venv_dir)
    cumm_arch_note = _cumm_arch_override_note(gpu_sm)
    return (
        NativeInstallStep(
            package="native-build-prerequisites",
            requirement="pccm>=0.4.16 ccimport>=0.4.4 pybind11>=2.6.0 fire ninja cmake",
            strategy="pip-install-prerequisites",
            status="planned",
            pip_args=(
                "install",
                "pccm>=0.4.16",
                "ccimport>=0.4.4",
                "pybind11>=2.6.0",
                "fire",
                "ninja",
                "cmake",
            ),
            env=common_env,
        ),
        NativeInstallStep(
            package="torch_scatter",
            requirement=f"torch-scatter=={TORCH_SCATTER_VERSION}",
            strategy="pip-source-build",
            status="planned",
            pip_args=("install", "--no-build-isolation", f"torch-scatter=={TORCH_SCATTER_VERSION}", "--no-cache-dir"),
            env=torch_pyg_env,
        ),
        NativeInstallStep(
            package="torch_cluster",
            requirement=f"torch-cluster=={TORCH_CLUSTER_VERSION}",
            strategy="pip-source-build",
            status="planned",
            pip_args=("install", "--no-build-isolation", f"torch-cluster=={TORCH_CLUSTER_VERSION}", "--no-cache-dir"),
            env=torch_pyg_env,
        ),
        NativeInstallStep(
            package="cumm",
            requirement="git+https://github.com/FindDefinition/cumm.git@v0.7.11",
            strategy="pip-git-source-build",
            status="planned",
            pip_args=(
                "install",
                "--no-build-isolation",
                "--no-deps",
                "git+https://github.com/FindDefinition/cumm.git@v0.7.11",
            ),
            env=cumm_env,
            reason=cumm_arch_note,
        ),
        NativeInstallStep(
            package="spconv",
            requirement="git+https://github.com/traveller59/spconv.git@v2.3.8",
            strategy="pip-git-source-build",
            status="planned",
            pip_args=(
                "install",
                "--no-build-isolation",
                "--no-deps",
                "git+https://github.com/traveller59/spconv.git@v2.3.8",
            ),
            env=spconv_env,
            reason=cumm_arch_note,
        ),
    )


def _windows_pyg_wheel_plan(torch_plan: dict[str, Any]) -> dict[str, str]:
    torch_version: str | None = None
    for requirement in torch_plan.get("pins") or torch_plan.get("packages") or ():
        match = re.match(r"^torch==(?P<version>\d+\.\d+\.\d+)(?:\+[^\s]+)?$", str(requirement))
        if match:
            torch_version = match.group("version")
            break
    if not torch_version:
        raise RuntimeError(f"Unable to derive Windows PyG wheel plan from torch packages: {torch_plan.get('packages')!r}.")

    cuda_label = str(torch_plan.get("label") or "").strip().lower()
    if not re.fullmatch(r"cu\d+", cuda_label):
        raise RuntimeError(f"Unable to derive Windows PyG CUDA suffix from torch plan label: {cuda_label!r}.")

    major, minor, _patch = torch_version.split(".", 2)
    return {
        "index_url": f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_label}.html",
        "suffix": f"pt{major}{minor}{cuda_label}",
        "torch_version": torch_version,
        "cuda_label": cuda_label,
    }


def _windows_amd64_native_install_steps(torch_plan: dict[str, Any]) -> tuple[NativeInstallStep, ...]:
    pyg_plan = _windows_pyg_wheel_plan(torch_plan)
    pyg_suffix = pyg_plan["suffix"]
    pyg_index_url = pyg_plan["index_url"]
    reason = (
        "Windows AMD64 uses PyG prebuilt wheels matching the managed PyTorch/CUDA install plan "
        f"({pyg_plan['torch_version']}+{pyg_plan['cuda_label']}) instead of source builds."
    )
    return (
        NativeInstallStep(
            package="torch_scatter",
            requirement=f"torch-scatter=={TORCH_SCATTER_VERSION}+{pyg_suffix}",
            strategy="pip-prebuilt-wheel",
            status="planned",
            pip_args=(
                "install",
                f"torch-scatter=={TORCH_SCATTER_VERSION}+{pyg_suffix}",
                "-f",
                pyg_index_url,
                "--no-cache-dir",
            ),
            reason=reason,
        ),
        NativeInstallStep(
            package="torch_cluster",
            requirement=f"torch-cluster=={TORCH_CLUSTER_VERSION}+{pyg_suffix}",
            strategy="pip-prebuilt-wheel",
            status="planned",
            pip_args=(
                "install",
                f"torch-cluster=={TORCH_CLUSTER_VERSION}+{pyg_suffix}",
                "-f",
                pyg_index_url,
                "--no-cache-dir",
            ),
            reason=reason,
        ),
        NativeInstallStep(
            package="spconv",
            requirement=WINDOWS_SPCONV_CU120_PACKAGE,
            strategy="pip-prebuilt-wheel",
            status="planned",
            pip_args=("install", WINDOWS_SPCONV_CU120_PACKAGE, "--no-cache-dir"),
            reason="spconv-cu120 publishes cp311/cp312 Windows AMD64 wheels and imports as the spconv module.",
        ),
    )


def _normalize_native_mode(value: Any) -> str:
    normalized = str(value or NATIVE_MODE_AUTO).strip().lower()
    if normalized not in SUPPORTED_NATIVE_MODES:
        raise RuntimeError(
            f"Unsupported native_mode={normalized!r}. Supported values: {', '.join(sorted(SUPPORTED_NATIVE_MODES))}."
        )
    return normalized


def _resolve_native_mode(payload: dict[str, Any] | None = None) -> str:
    source = payload or {}
    explicit = source.get("native_mode")
    if explicit not in (None, ""):
        return _normalize_native_mode(explicit)

    raw_args = list(source.get("args") or [])
    index = 0
    while index < len(raw_args):
        token = str(raw_args[index])
        if token.startswith("native_mode="):
            return _normalize_native_mode(token.split("=", 1)[1])
        if token.startswith("--native-mode="):
            return _normalize_native_mode(token.split("=", 1)[1])
        if token == "--native-mode" and index + 1 < len(raw_args):
            return _normalize_native_mode(raw_args[index + 1])
        index += 1

    for env_var in ("HUNYUAN3D_PART_NATIVE_MODE", "P3SAM_NATIVE_MODE"):
        env_value = os.environ.get(env_var)
        if env_value not in (None, ""):
            return _normalize_native_mode(env_value)
    return NATIVE_MODE_AUTO


def build_native_runtime_plan(
    *,
    system_name: str,
    machine: str,
    py_tag: str,
    gpu_sm: int,
    cuda_version: int,
    native_mode: str,
    venv_dir: Path | None = None,
) -> NativeRuntimePlan:
    resolved_mode = _normalize_native_mode(native_mode)
    normalized_machine = normalize_machine(machine)
    platform_key = f"{system_name.lower()}/{normalized_machine}"
    lane: str | None = None
    desired_cuda_label: str | None = None
    state = UNSUPPORTED
    install_strategy = "unsupported"
    reason = (
        "This extension currently validates explicit P3-SAM native dependency planning only for Linux ARM64 "
        f"lanes. Received unsupported native host {platform_key}."
    )
    if _is_linux_arm64(system_name, normalized_machine) and (gpu_sm >= 100 or cuda_version >= 128):
        lane = "linux-arm64-cu128"
        desired_cuda_label = "cu128"
        state = SUPPORTED
        reason = "Linux ARM64 CUDA 12.8 lane matches the validated base-runtime torch selection for this extension."
    elif _is_linux_arm64(system_name, normalized_machine) and gpu_sm >= 70:
        lane = "linux-arm64-cu124"
        desired_cuda_label = "cu124"
        state = SUPPORTED
        reason = "Linux ARM64 CUDA 12.4 lane matches the validated base-runtime torch selection for this extension."
    elif _is_linux_arm64(system_name, normalized_machine):
        reason = (
            "Linux ARM64 native dependency planning requires an NVIDIA CUDA host with gpu_sm >= 70 for the "
            f"validated base-runtime lanes. Received gpu_sm={gpu_sm} cuda_version={cuda_version}."
        )
    elif _is_windows_amd64(system_name, normalized_machine):
        torch_plan = torch_install_plan(
            system_name=system_name,
            machine=normalized_machine,
            py_tag=py_tag,
            gpu_sm=gpu_sm,
            cuda_version=cuda_version,
        )
        desired_cuda_label = str(torch_plan.get("label") or "")
        lane = f"windows-amd64-{desired_cuda_label}-prebuilt"
        state = SUPPORTED
        reason = (
            "Windows AMD64 uses prebuilt PyG wheels matching the selected PyTorch/CUDA lane plus spconv-cu120; "
            "runtime inference still depends on the managed import smoke passing on the target host."
        )

    host = NativeHostLane(
        system=system_name,
        machine=normalized_machine,
        platform_key=platform_key,
        python_tag=py_tag,
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
        lane=lane,
        state=state,
        reason=reason,
    )
    toolkit = resolve_cuda_toolkit_for_label(desired_cuda_label)
    resolved_venv_dir = venv_dir or (ROOT / "venv")
    if state == SUPPORTED and _is_windows_amd64(system_name, normalized_machine):
        install_strategy = "windows-amd64-prebuilt-wheels"
        torch_plan = torch_install_plan(
            system_name=system_name,
            machine=normalized_machine,
            py_tag=py_tag,
            gpu_sm=gpu_sm,
            cuda_version=cuda_version,
        )
        install_steps = _windows_amd64_native_install_steps(torch_plan)
        message = (
            "Windows AMD64 native runtime uses prebuilt wheels and may be installed automatically during managed setup."
        )
        next_action = "Run managed setup/repair again; it will install Windows prebuilt native wheels and re-probe readiness."
        plan_status = "planned"
        automatic_install_supported = True
    elif state == SUPPORTED:
        install_strategy = "linux-arm64-source-build"
        install_steps = _linux_arm64_native_install_steps(gpu_sm=gpu_sm, toolkit=toolkit, venv_dir=resolved_venv_dir) if toolkit["status"] == "matched" else (
            NativeInstallStep(
                package="native-build-prerequisites",
                requirement="pccm>=0.4.16 ccimport>=0.4.4 pybind11>=2.6.0 fire ninja cmake",
                strategy="pip-install-prerequisites",
                status="blocked",
                reason=toolkit["reason"],
            ),
            NativeInstallStep(
                package="torch_scatter",
                requirement=f"torch-scatter=={TORCH_SCATTER_VERSION}",
                strategy="pip-source-build",
                status="blocked",
                reason=toolkit["reason"],
            ),
            NativeInstallStep(
                package="torch_cluster",
                requirement=f"torch-cluster=={TORCH_CLUSTER_VERSION}",
                strategy="pip-source-build",
                status="blocked",
                reason=toolkit["reason"],
            ),
            NativeInstallStep(
                package="cumm",
                requirement="git+https://github.com/FindDefinition/cumm.git@v0.7.11",
                strategy="pip-git-source-build",
                status="blocked",
                reason=toolkit["reason"],
            ),
            NativeInstallStep(
                package="spconv",
                requirement="git+https://github.com/traveller59/spconv.git@v2.3.8",
                strategy="pip-git-source-build",
                status="blocked",
                reason=toolkit["reason"],
            ),
        )
        if toolkit["status"] == "matched":
            message = (
                "Explicit native runtime phase stays safe in auto mode and exposes a guarded Linux ARM64 source-build "
                "repair path only when native_mode=install is explicitly requested."
            )
            next_action = (
                "Run native_mode=install only after explicit confirmation to attempt the guarded source-build repair path."
            )
            plan_status = "planned"
            automatic_install_supported = True
        else:
            message = "Explicit native runtime repair is blocked until a CUDA toolkit matching the selected PyTorch lane is available."
            next_action = toolkit["reason"]
            plan_status = "blocked"
            automatic_install_supported = False
    else:
        install_steps = (
            NativeInstallStep(
                package="torch_scatter",
                requirement="torch_scatter",
                strategy="unvalidated-package-strategy",
                status="blocked",
                reason="No validated wheel/index strategy is encoded for this extension lane.",
            ),
            NativeInstallStep(
                package="torch_cluster",
                requirement="torch_cluster",
                strategy="unvalidated-package-strategy",
                status="blocked",
                reason="No validated wheel/index strategy is encoded for this extension lane.",
            ),
            NativeInstallStep(
                package="cumm",
                requirement="cumm",
                strategy="unvalidated-package-strategy",
                status="blocked",
                reason="No validated wheel/index strategy is encoded for this extension lane.",
            ),
            NativeInstallStep(
                package="spconv",
                requirement="spconv",
                strategy="unvalidated-package-strategy",
                status="blocked",
                reason="No validated wheel/index strategy is encoded for this extension lane.",
            ),
        )
        message = (
            "Explicit native runtime phase is plan-and-probe only for unsupported lanes; guarded install is limited "
            "to validated Linux ARM64 source-build paths."
        )
        next_action = reason
        plan_status = "unsupported"
        automatic_install_supported = False
    return NativeRuntimePlan(
        native_mode=resolved_mode,
        host=host,
        desired_cuda_label=toolkit["desired_cuda_label"],
        toolkit_root=toolkit["toolkit_root"],
        nvcc_version=toolkit["nvcc_version"],
        toolkit_status=toolkit["status"],
        toolkit_reason=toolkit["reason"],
        required_imports=_native_required_imports(),
        related_packages=_native_related_packages(),
        install_steps=install_steps,
        install_strategy=install_strategy,
        automatic_install_supported=automatic_install_supported,
        status=plan_status,
        message=message,
        next_action=next_action,
        toolkit_include_dir=toolkit.get("include_dir"),
        toolkit_nvcc=toolkit.get("nvcc_path"),
        toolkit_lib_dirs=tuple(toolkit.get("lib_dirs") or ()),
    )


def probe_native_runtime(*, managed_python: Path, plan: NativeRuntimePlan) -> NativeRuntimeProbe:
    if not managed_python.exists():
        return NativeRuntimeProbe(
            ready=False,
            status="pending",
            managed_python=str(managed_python),
            imports={},
            missing_modules=(),
            blocked_imports=(),
            message="Managed Python is missing, so native runtime imports cannot be probed yet.",
            next_action="Run managed setup/repair to create the extension venv before probing native dependencies.",
        )

    script = (
        "import importlib, json; "
        f"mods={list(_native_probe_modules())!r}; "
        "payload={}; "
        "\nfor name in mods:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "        payload[name]={'ready': True, 'status': 'ready'}\n"
        "    except Exception as exc:\n"
        "        payload[name]={'ready': False, 'status': 'missing', 'error_type': type(exc).__name__, 'error': str(exc)}\n"
        "print(json.dumps(payload, sort_keys=True))"
    )
    result = subprocess.run([str(managed_python), "-c", script], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return NativeRuntimeProbe(
            ready=False,
            status="blocked",
            managed_python=str(managed_python),
            imports={},
            missing_modules=_native_probe_modules(),
            blocked_imports=plan.required_imports,
            message="Native runtime probe failed before import status could be decoded.",
            next_action="Inspect the managed Python stderr and repair the extension venv before retrying.",
        )

    imports = json.loads(result.stdout.strip() or "{}")
    missing_modules = tuple(name for name, status in imports.items() if not status.get("ready", False))
    blocked_imports = tuple(name for name in plan.required_imports if name in missing_modules)
    ready = not blocked_imports
    message = (
        "Native runtime imports are ready."
        if ready
        else "Native runtime imports are blocked; adapter import smoke will fail closed until required modules are present."
    )
    next_action = (
        "None. Native runtime imports are present."
        if ready
        else "Provide a validated native dependency install path or repair the managed venv manually for the reported lane."
    )
    return NativeRuntimeProbe(
        ready=ready,
        status="ready" if ready else "blocked",
        managed_python=str(managed_python),
        imports=imports,
        missing_modules=missing_modules,
        blocked_imports=blocked_imports,
        message=message,
        next_action=next_action,
    )


def _tail_lines(text: str, *, limit: int = 20) -> list[str]:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    return lines[-limit:]


def _build_error_lines(*texts: str, limit: int = 20) -> list[str]:
    patterns = (
        r"(^|\s)error:",
        r"fatal error:",
        r"undefined reference",
        r"collect2:",
        r"ld:",
        r"ninja: build stopped",
        r"returned non-zero exit status",
    )
    combined = re.compile("|".join(patterns), re.IGNORECASE)
    matched: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for raw_line in (text or "").splitlines():
            line = raw_line.rstrip()
            if not line.strip() or not combined.search(line):
                continue
            if line in seen:
                continue
            seen.add(line)
            matched.append(line)
    return matched[-limit:]


def probe_cumm_core_cc(managed_python: Path) -> dict[str, Any]:
    module_name = "cumm.core_cc"
    if not managed_python.exists():
        return {
            "package": "cumm-core-cc",
            "module": module_name,
            "ready": False,
            "status": "pending",
            "managed_python": str(managed_python),
            "error": f"Managed Python is missing: {managed_python}",
            "error_type": "FileNotFoundError",
            "traceback_tail": [],
            "next_action": "Run managed setup/repair to create the extension venv before validating cumm.core_cc.",
        }

    script = (
        "import importlib, json, traceback; "
        f"module={module_name!r}; "
        "payload={'package': 'cumm-core-cc', 'module': module, 'ready': False, 'status': 'blocked', "
        "'managed_python': __import__('sys').executable, 'error': None, 'error_type': None, 'traceback_tail': [], "
        "'next_action': 'Inspect the managed cumm install and ensure cumm.core_cc is built before attempting spconv.'}; "
        "\ntry:\n"
        "    importlib.import_module('cumm')\n"
        "    importlib.import_module(module)\n"
        "except Exception as exc:\n"
        "    payload['error'] = str(exc)\n"
        "    payload['error_type'] = type(exc).__name__\n"
        "    payload['status'] = 'missing' if isinstance(exc, ModuleNotFoundError) else 'blocked'\n"
        "    payload['traceback_tail'] = traceback.format_exc().splitlines()[-20:]\n"
        "else:\n"
        "    payload.update({'ready': True, 'status': 'ready', 'next_action': 'None. cumm.core_cc imports successfully.'})\n"
        "print(json.dumps(payload, sort_keys=True))"
    )
    result = subprocess.run([str(managed_python), "-c", script], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "package": "cumm-core-cc",
            "module": module_name,
            "ready": False,
            "status": "failed",
            "managed_python": str(managed_python),
            "error": "Managed Python failed before the cumm.core_cc validation payload could be decoded.",
            "error_type": "SubprocessError",
            "traceback_tail": _tail_lines(result.stderr or result.stdout),
            "next_action": "Inspect the managed Python stderr/stdout and repair the cumm install before attempting spconv.",
        }
    try:
        payload = json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {
            "package": "cumm-core-cc",
            "module": module_name,
            "ready": False,
            "status": "failed",
            "managed_python": str(managed_python),
            "error": "Managed Python returned a non-JSON cumm.core_cc validation payload.",
            "error_type": "JSONDecodeError",
            "traceback_tail": _tail_lines(result.stdout or result.stderr),
            "next_action": "Inspect the managed Python output and repair the cumm install before attempting spconv.",
        }
    payload["package"] = payload.get("package") or "cumm-core-cc"
    payload["module"] = payload.get("module") or module_name
    payload["managed_python"] = payload.get("managed_python") or str(managed_python)
    payload["traceback_tail"] = list(payload.get("traceback_tail") or [])
    return payload


def probe_spconv_core_cc(managed_python: Path) -> dict[str, Any]:
    module_name = "spconv.core_cc"
    if not managed_python.exists():
        return {
            "package": "spconv-core-cc",
            "module": module_name,
            "ready": False,
            "status": "pending",
            "managed_python": str(managed_python),
            "error": f"Managed Python is missing: {managed_python}",
            "error_type": "FileNotFoundError",
            "traceback_tail": [],
            "next_action": "Run managed setup/repair to create the extension venv before validating spconv.core_cc.",
        }

    script = (
        "import importlib, json, traceback; "
        f"module={module_name!r}; "
        "payload={'package': 'spconv-core-cc', 'module': module, 'ready': False, 'status': 'blocked', "
        "'managed_python': __import__('sys').executable, 'error': None, 'error_type': None, 'traceback_tail': [], "
        "'next_action': 'Inspect the managed spconv install and rebuild with SPCONV_DISABLE_JIT=1 so spconv.core_cc is produced during install.'}; "
        "\ntry:\n"
        "    importlib.import_module('spconv')\n"
        "    importlib.import_module(module)\n"
        "except Exception as exc:\n"
        "    payload['error'] = str(exc)\n"
        "    payload['error_type'] = type(exc).__name__\n"
        "    payload['status'] = 'missing' if isinstance(exc, ModuleNotFoundError) else 'blocked'\n"
        "    payload['traceback_tail'] = traceback.format_exc().splitlines()[-20:]\n"
        "else:\n"
        "    payload.update({'ready': True, 'status': 'ready', 'next_action': 'None. spconv.core_cc imports successfully.'})\n"
        "print(json.dumps(payload, sort_keys=True))"
    )
    result = subprocess.run([str(managed_python), "-c", script], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "package": "spconv-core-cc",
            "module": module_name,
            "ready": False,
            "status": "failed",
            "managed_python": str(managed_python),
            "error": "Managed Python failed before the spconv.core_cc validation payload could be decoded.",
            "error_type": "SubprocessError",
            "traceback_tail": _tail_lines(result.stderr or result.stdout),
            "next_action": "Inspect the managed Python stderr/stdout and repair the spconv install before adapter smoke.",
        }
    try:
        payload = json.loads(result.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {
            "package": "spconv-core-cc",
            "module": module_name,
            "ready": False,
            "status": "failed",
            "managed_python": str(managed_python),
            "error": "Managed Python returned a non-JSON spconv.core_cc validation payload.",
            "error_type": "JSONDecodeError",
            "traceback_tail": _tail_lines(result.stdout or result.stderr),
            "next_action": "Inspect the managed Python output and repair the spconv install before adapter smoke.",
        }
    payload["package"] = payload.get("package") or "spconv-core-cc"
    payload["module"] = payload.get("module") or module_name
    payload["managed_python"] = payload.get("managed_python") or str(managed_python)
    payload["traceback_tail"] = list(payload.get("traceback_tail") or [])
    return payload


def _native_install_result_succeeded(result: dict[str, Any]) -> bool:
    return result.get("status") in {"installed", "already_present", "already_patched", "patched", "repaired", "ready"}


def _venv_site_packages_candidates(venv_dir: Path) -> tuple[Path, ...]:
    if platform.system() == "Windows":
        return (venv_dir / "Lib" / "site-packages",)
    return tuple(sorted((venv_dir / "lib").glob("python*/site-packages")))


def _valid_cumm_include_dir(path: Path) -> bool:
    return path.is_dir() and (path / CUMM_TENSORVIEW_SENTINEL).is_file()


def inspect_cumm_tensorview_layout(venv_dir: Path) -> dict[str, Any]:
    site_packages_candidates = _venv_site_packages_candidates(venv_dir)
    constants_path: Path | None = None
    site_packages: Path | None = None
    for candidate in site_packages_candidates:
        current = candidate / "cumm" / "constants.py"
        if current.is_file():
            constants_path = current
            site_packages = candidate
            break

    if constants_path is None or site_packages is None:
        fallback_site_packages = site_packages_candidates[0] if site_packages_candidates else (venv_dir / "lib")
        return {
            "status": "missing",
            "package": "cumm-tensorview-headers",
            "venv_dir": str(venv_dir),
            "site_packages": str(fallback_site_packages),
            "next_action": "Install cumm into the managed venv before attempting tensorview header repair.",
        }

    package_root = constants_path.parent.resolve()
    include_paths = [package_root.parent / "include", package_root / "include"]
    entries = [
        {
            "path": str(path),
            "exists": path.exists(),
            "ready": _valid_cumm_include_dir(path),
        }
        for path in include_paths
    ]
    active_path = next((Path(entry["path"]) for entry in entries if entry["exists"]), include_paths[0])
    ready_path = next((Path(entry["path"]) for entry in entries if entry["ready"]), None)
    return {
        "status": "ready" if ready_path else "blocked",
        "package": "cumm-tensorview-headers",
        "venv_dir": str(venv_dir),
        "site_packages": str(site_packages),
        "constants_path": str(constants_path),
        "package_root": str(package_root),
        "expected_include_path": str(include_paths[0]),
        "fallback_include_path": str(include_paths[1]),
        "active_include_path": str(ready_path or active_path),
        "include_candidates": entries,
        "next_action": "None. tensorview headers are available." if ready_path else "Repair or provide the missing cumm tensorview headers before importing spconv.",
    }


def _iter_cumm_header_source_candidates(search_roots: tuple[Path, ...]) -> tuple[dict[str, Any], ...]:
    discovered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in search_roots:
        if not root.exists():
            continue
        direct_include = root / "include"
        if _valid_cumm_include_dir(direct_include):
            resolved = str(direct_include.resolve())
            if resolved not in seen:
                seen.add(resolved)
                discovered.append({"path": resolved, "source": str(root), "kind": "existing-include"})
        for sentinel in root.glob("**/include/tensorview/tensor.h"):
            include_dir = sentinel.parent.parent
            if not _valid_cumm_include_dir(include_dir):
                continue
            resolved = str(include_dir.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append({"path": resolved, "source": str(root), "kind": "discovered-include"})
    return tuple(discovered)


def prepare_cumm_source_headers(
    extension_dir: Path,
    *,
    repo_url: str = CUMM_REPO_URL,
    repo_ref: str = CUMM_REPO_REF,
) -> dict[str, Any]:
    source_root = extension_dir / ".cache" / "native-sources" / f"cumm-{repo_ref}"
    include_dir = source_root / "include"
    if _valid_cumm_include_dir(include_dir):
        return {
            "status": "ready",
            "source_root": str(source_root),
            "include_path": str(include_dir),
            "action": "reuse",
            "next_action": "None. Cached cumm headers are ready.",
        }

    if source_root.exists():
        shutil.rmtree(source_root)
    source_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--sparse",
            "--depth",
            "1",
            "--branch",
            repo_ref,
            repo_url,
            str(source_root),
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(source_root), "sparse-checkout", "set", "include"], check=True)
    if not _valid_cumm_include_dir(include_dir):
        return {
            "status": "failed",
            "source_root": str(source_root),
            "include_path": str(include_dir),
            "action": "clone",
            "next_action": "Inspect the cached cumm checkout because the include/tensorview headers were still missing after clone.",
        }
    return {
        "status": "ready",
        "source_root": str(source_root),
        "include_path": str(include_dir),
        "action": "clone",
        "next_action": "None. Cached cumm headers are ready.",
    }


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def repair_cumm_tensorview_headers(
    venv_dir: Path,
    *,
    extension_dir: Path | None = None,
    source_search_roots: tuple[Path, ...] | None = None,
    source_preparer: Any = prepare_cumm_source_headers,
) -> dict[str, Any]:
    extension_dir = (extension_dir or venv_dir.parent).resolve()
    layout = inspect_cumm_tensorview_layout(venv_dir)
    result: dict[str, Any] = {
        "package": "cumm-tensorview-headers",
        "status": layout["status"],
        "venv_dir": str(venv_dir),
        "expected_path": layout.get("expected_include_path"),
        "active_path": layout.get("active_include_path"),
        "constants_path": layout.get("constants_path"),
        "source_path": None,
        "source_kind": None,
        "action": "inspect",
        "attempted": False,
        "next_action": layout.get("next_action"),
    }
    if layout["status"] == "missing":
        result["status"] = "blocked"
        return result
    if layout["status"] == "ready":
        result["status"] = "already_present"
        result["action"] = "noop"
        result["next_action"] = "None. tensorview headers are already present."
        return result

    expected_path = Path(layout["expected_include_path"])
    search_roots = source_search_roots or (
        extension_dir / ".cache" / "native-sources",
        extension_dir,
        venv_dir,
        Path.home() / ".cache" / "pip",
    )
    candidates = list(_iter_cumm_header_source_candidates(tuple(search_roots)))
    source_candidate: dict[str, Any] | None = candidates[0] if candidates else None
    if source_candidate is None:
        try:
            prepared = source_preparer(extension_dir)
        except Exception as exc:
            prepared = {
                "status": "failed",
                "action": "clone",
                "reason": str(exc),
                "error_type": type(exc).__name__,
                "next_action": "Inspect the cumm source fetch failure and retry the managed repair after network/source access is restored.",
            }
        if prepared.get("status") == "ready" and _valid_cumm_include_dir(Path(prepared["include_path"])):
            source_candidate = {
                "path": str(Path(prepared["include_path"]).resolve()),
                "source": prepared.get("source_root") or prepared["include_path"],
                "kind": f"prepared-{prepared.get('action', 'source')}",
            }
        else:
            result.update(
                {
                    "status": "failed",
                    "action": prepared.get("action", "clone"),
                    "attempted": True,
                    "reason": prepared.get("reason", "Unable to locate cumm tensorview headers in the managed venv, caches, or prepared source checkout."),
                    "error_type": prepared.get("error_type"),
                    "next_action": prepared.get(
                        "next_action",
                        "Provide a valid cumm include/tensorview header source and retry native_mode=install.",
                    ),
                }
            )
            return result

    source_path = Path(source_candidate["path"])
    if not _valid_cumm_include_dir(source_path):
        result.update(
            {
                "status": "failed",
                "attempted": False,
                "reason": f"Located candidate include path is invalid: {source_path}",
                "source_path": str(source_path),
                "source_kind": source_candidate.get("kind"),
                "next_action": "Provide a valid cumm include/tensorview tree and retry the managed repair.",
            }
        )
        return result

    if expected_path.resolve() == source_path.resolve():
        result.update(
            {
                "status": "already_present",
                "source_path": str(source_path),
                "source_kind": source_candidate.get("kind"),
                "action": "noop",
                "next_action": "None. tensorview headers are already present at the expected path.",
            }
        )
        return result

    expected_path.parent.mkdir(parents=True, exist_ok=True)
    if expected_path.exists() or expected_path.is_symlink():
        _remove_path(expected_path)
    attempted = True
    action = "symlink"
    try:
        expected_path.symlink_to(source_path, target_is_directory=True)
    except OSError:
        shutil.copytree(source_path, expected_path)
        action = "copy"

    repaired = _valid_cumm_include_dir(expected_path)
    result.update(
        {
            "status": "repaired" if repaired else "failed",
            "source_path": str(source_path),
            "source_kind": source_candidate.get("kind"),
            "action": action,
            "attempted": attempted,
            "active_path": str(expected_path),
            "next_action": "None. tensorview headers were repaired for cumm." if repaired else "Inspect the managed include repair output and retry native_mode=install.",
        }
    )
    if not repaired:
        result["reason"] = f"Expected include path is still incomplete after {action}: {expected_path}"
    return result


def patch_cumm_subint_header(venv_dir: Path) -> dict[str, Any]:
    layout = inspect_cumm_tensorview_layout(venv_dir)
    result: dict[str, Any] = {
        "package": "cumm-subint-header-patch",
        "status": "blocked",
        "venv_dir": str(venv_dir),
        "include_path": layout.get("active_include_path"),
        "path": None,
        "action": "inspect",
        "attempted": False,
        "next_action": layout.get("next_action"),
    }
    if layout.get("status") != "ready":
        result["reason"] = "cumm include path is not ready for subint header patching"
        return result

    include_path = Path(layout["active_include_path"])
    header_path = include_path / CUMM_SUBINT_HEADER_RELATIVE_PATH
    result["path"] = str(header_path)
    if not header_path.is_file():
        result["reason"] = "missing_subint_header"
        result["next_action"] = "Repair or reinstall cumm so tensorview/gemm/dtypes/subint.h exists before attempting spconv."
        return result

    source = header_path.read_text(encoding="utf-8")
    split_header = _split_cumm_subint_header_sections(source)
    if split_header is None:
        result["status"] = "failed"
        result["reason"] = "unknown_subint_header_source"
        result["next_action"] = "Inspect the installed cumm subint header manually because the guarded patch only supports the known upstream header layout."
        return result

    prefix, body, suffix = split_header
    normalized_body = _normalize_subint_body_for_match(body)
    if normalized_body == _normalize_subint_body_for_match(CUMM_SUBINT_PATCH_BODY_REPLACEMENT):
        result["status"] = "already_patched"
        result["action"] = "noop"
        result["next_action"] = "None. cumm subint header patch is already present."
        return result

    if normalized_body in {
        _normalize_subint_body_for_match(CUMM_SUBINT_PATCH_BODY_ORIGINAL),
        _normalize_subint_body_for_match(CUMM_SUBINT_PATCH_BODY_REPLACEMENT_LEGACY),
    }:
        patched_source = prefix + CUMM_SUBINT_PATCH_BODY_REPLACEMENT + suffix
    elif _is_known_malformed_cumm_subint_body(body):
        patched_source = prefix + CUMM_SUBINT_PATCH_BODY_REPLACEMENT + suffix
    else:
        result["status"] = "failed"
        result["reason"] = "unknown_subint_header_source"
        result["next_action"] = "Inspect the installed cumm subint header manually because the guarded patch only supports the known upstream body or the known malformed patched variant."
        return result

    header_path.write_text(patched_source, encoding="utf-8")
    verified = header_path.read_text(encoding="utf-8")
    result["attempted"] = True
    result["action"] = "rewrite"
    verified_split = _split_cumm_subint_header_sections(verified)
    if verified_split is None or verified_split[1] != CUMM_SUBINT_PATCH_BODY_REPLACEMENT:
        result["status"] = "failed"
        result["reason"] = "subint_header_patch_verification_failed"
        result["next_action"] = "Inspect the rewritten cumm subint header because the guarded patch could not be verified."
        return result

    result["status"] = "patched"
    result["next_action"] = "None. guarded cumm subint header patch is in place."
    return result


def patch_cumm_half_header(venv_dir: Path) -> dict[str, Any]:
    layout = inspect_cumm_tensorview_layout(venv_dir)
    result: dict[str, Any] = {
        "package": "cumm-half-header-patch",
        "status": "blocked",
        "venv_dir": str(venv_dir),
        "include_path": layout.get("active_include_path"),
        "path": None,
        "action": "inspect",
        "attempted": False,
        "next_action": layout.get("next_action"),
    }
    if layout.get("status") != "ready":
        result["reason"] = "cumm include path is not ready for half header patching"
        return result

    include_path = Path(layout["active_include_path"])
    header_path = include_path / CUMM_HALF_HEADER_RELATIVE_PATH
    result["path"] = str(header_path)
    if not header_path.is_file():
        result["reason"] = "missing_half_header"
        result["next_action"] = "Repair or reinstall cumm so tensorview/gemm/dtypes/half.h exists before attempting spconv."
        return result

    source = header_path.read_text(encoding="utf-8")
    patched_source, patch_status, patch_reason = _canonicalize_cumm_half_header(source)
    if patch_reason is not None:
        result["status"] = "failed"
        result["reason"] = patch_reason
        result["next_action"] = (
            "Inspect the installed cumm half header manually because the guarded patch only supports the known live half.h layout around the cuda_fp16 include section."
        )
        return result
    if patch_status == "already_patched":
        result["status"] = "already_patched"
        result["action"] = "noop"
        result["next_action"] = "None. guarded cumm half header patch is already present."
        return result

    header_path.write_text(patched_source, encoding="utf-8")
    verified_source = header_path.read_text(encoding="utf-8")
    result["attempted"] = True
    result["action"] = "rewrite"
    verified_patched_source, verified_status, verified_reason = _canonicalize_cumm_half_header(verified_source)
    if verified_reason is not None or verified_status != "already_patched" or verified_patched_source != verified_source:
        result["status"] = "failed"
        result["reason"] = "half_header_patch_verification_failed"
        result["next_action"] = "Inspect the rewritten cumm half header because the guarded patch could not be verified."
        return result

    result["status"] = "patched"
    result["next_action"] = "None. guarded cumm half header patch is in place."
    return result


def patch_cumm_bfloat16_header(venv_dir: Path) -> dict[str, Any]:
    layout = inspect_cumm_tensorview_layout(venv_dir)
    result: dict[str, Any] = {
        "package": "cumm-bfloat16-header-patch",
        "status": "blocked",
        "venv_dir": str(venv_dir),
        "include_path": layout.get("active_include_path"),
        "path": None,
        "action": "inspect",
        "attempted": False,
        "next_action": layout.get("next_action"),
    }
    if layout.get("status") != "ready":
        result["reason"] = "cumm include path is not ready for bfloat16 header patching"
        return result

    include_path = Path(layout["active_include_path"])
    header_path = include_path / CUMM_BFLOAT16_HEADER_RELATIVE_PATH
    result["path"] = str(header_path)
    if not header_path.is_file():
        result["reason"] = "missing_bfloat16_header"
        result["next_action"] = "Repair or reinstall cumm so tensorview/gemm/dtypes/bfloat16.h exists before attempting spconv."
        return result

    source = header_path.read_text(encoding="utf-8")
    patched_source, patch_status, patch_reason = _canonicalize_cumm_bfloat16_header(source)
    if patch_reason is not None:
        result["status"] = "failed"
        result["reason"] = patch_reason
        result["next_action"] = (
            "Inspect the installed cumm bfloat16 header manually because the guarded patch only supports the known live bfloat16.h layout around the cuda_bf16 include section."
        )
        return result
    if patch_status == "already_patched":
        result["status"] = "already_patched"
        result["action"] = "noop"
        result["next_action"] = "None. guarded cumm bfloat16 header patch is already present."
        return result

    header_path.write_text(patched_source, encoding="utf-8")
    verified_source = header_path.read_text(encoding="utf-8")
    result["attempted"] = True
    result["action"] = "rewrite"
    verified_patched_source, verified_status, verified_reason = _canonicalize_cumm_bfloat16_header(verified_source)
    if verified_reason is not None or verified_status != "already_patched" or verified_patched_source != verified_source:
        result["status"] = "failed"
        result["reason"] = "bfloat16_header_patch_verification_failed"
        result["next_action"] = "Inspect the rewritten cumm bfloat16 header because the guarded patch could not be verified."
        return result

    result["status"] = "patched"
    result["next_action"] = "None. guarded cumm bfloat16 header patch is in place."
    return result


def _canonicalize_cumm_half_header(source: str) -> tuple[str, str, str | None]:
    target_source = _inject_cumm_host_math_macro_undef_block(
        source,
        anchor=CUMM_HALF_INCLUDE_SECTION,
        names=CUMM_HOST_MATH_MACRO_NAMES,
    )
    if target_source is None:
        return source, "failed", "unknown_half_header_source"
    target_source = _canonicalize_cumm_half_convert_member(target_source)
    if target_source is None:
        return source, "failed", "unknown_half_header_source"
    target_source = _canonicalize_cumm_math_namespace_free_function_region(
        target_source,
        struct_name="half_t",
        scalar_type="tv::half_t",
        copysign_signature=(
            "static TV_HOST_DEVICE_INLINE\ntv::half_t copysign(tv::half_t const &a, tv::half_t const &b) {",
            "TV_HOST_DEVICE_INLINE\nhalf_t copysign(half_t const &a, half_t const &b) {",
            "TV_HOST_DEVICE_INLINE\ntv::half_t copysign(tv::half_t const &a, tv::half_t const &b) {",
            "static TV_HOST_DEVICE_INLINE\nhalf_t copysign(half_t const &a, half_t const &b) {",
        ),
    )
    if target_source is None:
        return source, "failed", "unknown_half_header_source"
    if target_source == source:
        return source, "already_patched", None
    return target_source, "patched", None


def _canonicalize_cumm_bfloat16_header(source: str) -> tuple[str, str, str | None]:
    target_source = _inject_cumm_host_math_macro_undef_block(
        source,
        anchor="#include <tensorview/core/all.h>\n",
        names=CUMM_HOST_MATH_MACRO_NAMES,
    )
    if target_source is None:
        return source, "failed", "unknown_bfloat16_header_source"
    target_source = _canonicalize_cumm_math_namespace_free_function_region(
        target_source,
        struct_name="bfloat16_t",
        scalar_type="tv::bfloat16_t",
        copysign_signature=(
            "static TV_HOST_DEVICE_INLINE\ntv::bfloat16_t copysign(tv::bfloat16_t const &a, tv::bfloat16_t const &b) {",
            "TV_HOST_DEVICE_INLINE\nbfloat16_t copysign(bfloat16_t const &a, bfloat16_t const &b) {",
            "TV_HOST_DEVICE_INLINE\ntv::bfloat16_t copysign(tv::bfloat16_t const &a, tv::bfloat16_t const &b) {",
            "static TV_HOST_DEVICE_INLINE\nbfloat16_t copysign(bfloat16_t const &a, bfloat16_t const &b) {",
        ),
        drop_signbit=True,
    )
    if target_source is None:
        return source, "failed", "unknown_bfloat16_header_source"
    if target_source == source:
        return source, "already_patched", None
    return target_source, "patched", None


def _split_cumm_subint_header_sections(source: str) -> tuple[str, str, str] | None:
    body_start = source.find(CUMM_SUBINT_BODY_START_MARKER)
    body_end = source.rfind(CUMM_SUBINT_BODY_SUFFIX_MARKER)
    if body_start == -1 or body_end == -1 or body_end <= body_start:
        return None
    return source[:body_start], source[body_start:body_end], source[body_end:]


def _normalize_subint_body_for_match(body: str) -> str:
    if body.endswith("\n"):
        return body[:-1]
    return body


def _is_known_malformed_cumm_subint_body(body: str) -> bool:
    normalized_body = _normalize_subint_body_for_match(body)
    patch_signatures = (
        CUMM_SUBINT_PATCH_REPLACEMENT_LEGACY,
        CUMM_SUBINT_PATCH_REPLACEMENT,
    )
    if not any(signature in normalized_body for signature in patch_signatures):
        return False
    if normalized_body == _normalize_subint_body_for_match(CUMM_SUBINT_PATCH_BODY_REPLACEMENT):
        return False
    if normalized_body.count("template <int Bits, bool Signed = true> struct integer_subbyte {") != 1:
        return False
    if normalized_body.count("using uint1b_t = integer_subbyte<1, false>;") != 1:
        return False
    if normalized_body.count("template <> struct sizeof_bits<uint4b_t> { static int const value = 4; };") != 1:
        return False

    struct_end = normalized_body.find("\n};")
    if struct_end == -1:
        return False
    trailing_after_struct = normalized_body[struct_end + 3 :]
    malformed_tokens = (
        "T value() const {",
        "explicit operator T() const { return value(); }",
        "operator T() const {",
        "bool operator==(self_type const &rhs) const { return storage == rhs.storage; }",
        "bool operator!=(self_type const &rhs) const { return storage != rhs.storage; }",
        "bool operator<=(self_type const &rhs) const { return value() <= rhs.value(); }",
        "bool operator<(self_type const &rhs) const { return value() < rhs.value(); }",
        "bool operator>=(self_type const &rhs) const { return value() >= rhs.value(); }",
        "bool operator>(self_type const &rhs) const { return value() > rhs.value(); }",
        "bool operator<=(self_type const &rhs) const { return static_cast<T>(*this) <= static_cast<T>(rhs); }",
        "bool operator<(self_type const &rhs) const { return static_cast<T>(*this) < static_cast<T>(rhs); }",
        "bool operator>=(self_type const &rhs) const { return static_cast<T>(*this) >= static_cast<T>(rhs); }",
        "bool operator>(self_type const &rhs) const { return static_cast<T>(*this) > static_cast<T>(rhs); }",
    )
    return any(token in trailing_after_struct for token in malformed_tokens)


def patch_cumm_dtype_headers(venv_dir: Path) -> dict[str, Any]:
    layout = inspect_cumm_tensorview_layout(venv_dir)
    result: dict[str, Any] = {
        "package": "cumm-dtype-header-patch",
        "status": "blocked",
        "venv_dir": str(venv_dir),
        "include_path": layout.get("active_include_path"),
        "action": "inspect",
        "attempted": False,
        "headers": [],
        "next_action": layout.get("next_action"),
    }
    if layout.get("status") != "ready":
        result["reason"] = "cumm include path is not ready for dtype header patching"
        return result

    include_path = Path(layout["active_include_path"])
    header_specs = (
        {
            "name": "tf32",
            "relative_path": CUMM_TF32_HEADER_RELATIVE_PATH,
            "legacy_fragments": CUMM_TF32_HEADER_CANONICAL_FRAGMENTS,
        },
        {
            "name": "float8",
            "relative_path": CUMM_FLOAT8_HEADER_RELATIVE_PATH,
            "legacy_fragments": CUMM_FLOAT8_HEADER_CANONICAL_FRAGMENTS,
        },
    )
    planned_writes: list[tuple[Path, str]] = []
    header_results: list[dict[str, Any]] = []
    overall_status = "already_patched"
    for spec in header_specs:
        header_path = include_path / spec["relative_path"]
        header_result = {
            "name": spec["name"],
            "path": str(header_path),
            "status": "blocked",
        }
        if not header_path.is_file():
            header_result["status"] = "failed"
            header_result["reason"] = f"missing_{spec['name']}_header"
            result["headers"] = [*header_results, header_result]
            result["status"] = "failed"
            result["reason"] = header_result["reason"]
            result["next_action"] = (
                f"Repair or reinstall cumm so {spec['relative_path']} exists before attempting spconv."
            )
            return result

        source = header_path.read_text(encoding="utf-8")
        patched_source, patch_status, patch_reason = _canonicalize_cumm_dtype_header(
            name=spec["name"],
            relative_path=spec["relative_path"],
            legacy_fragments=spec["legacy_fragments"],
            source=source,
        )
        header_result["status"] = patch_status
        if patch_reason:
            header_result["reason"] = patch_reason
            result["headers"] = [*header_results, header_result]
            result["status"] = "failed"
            result["reason"] = f"unknown_{spec['name']}_header_source"
            result["next_action"] = (
                f"Inspect the installed cumm {spec['name']} header manually because the guarded patch only supports the repo-defined compiler-safe target plus known {CUMM_DTYPE_HEADER_LEGACY_VERSION}/{CUMM_DTYPE_HEADER_CANONICAL_VERSION} and local compatibility variants."
            )
            return result
        if patch_status == "patched":
            overall_status = "patched"
            planned_writes.append((header_path, patched_source))
        header_results.append(header_result)

    for header_path, patched_source in planned_writes:
        header_path.write_text(patched_source, encoding="utf-8")
    result["attempted"] = bool(planned_writes)
    result["action"] = "rewrite" if planned_writes else "noop"

    verified_headers: list[dict[str, Any]] = []
    for spec in header_specs:
        header_path = include_path / spec["relative_path"]
        verified_source = header_path.read_text(encoding="utf-8")
        target_source = _local_target_cumm_dtype_header_source(
            name=spec["name"],
            relative_path=spec["relative_path"],
            legacy_fragments=spec["legacy_fragments"],
        )
        verified_status = "already_patched" if verified_source == target_source else "failed"
        verified_reason = None if verified_status == "already_patched" else "unexpected_post_patch_content"
        verified_entry = {
            "name": spec["name"],
            "path": str(header_path),
            "status": verified_status,
        }
        if verified_reason:
            verified_entry["reason"] = verified_reason
            result["headers"] = verified_headers + [verified_entry]
            result["status"] = "failed"
            result["reason"] = f"{spec['name']}_header_patch_verification_failed"
            result["next_action"] = (
                f"Inspect the rewritten cumm {spec['name']} header because the guarded patch could not be verified."
            )
            return result
        verified_headers.append(verified_entry)

    result["headers"] = verified_headers
    result["status"] = overall_status
    result["next_action"] = "None. guarded cumm dtype header patch is in place."
    return result


def patch_cumm_cuda_resolver(
    venv_dir: Path,
    *,
    toolkit_root: str | None,
    include_dir: str | None,
    nvcc_path: str | None,
) -> dict[str, Any]:
    site_packages_candidates = _venv_site_packages_candidates(venv_dir)
    common_path: Path | None = None
    for candidate in site_packages_candidates:
        current = candidate / CUMM_COMMON_RELATIVE_PATH
        if current.is_file():
            common_path = current
            break

    result: dict[str, Any] = {
        "package": "cumm-cuda-resolver-patch",
        "status": "blocked",
        "venv_dir": str(venv_dir),
        "path": str(common_path) if common_path else None,
        "action": "inspect",
        "attempted": False,
        "selected_toolkit_root": toolkit_root,
        "selected_include_dir": include_dir,
        "selected_nvcc_path": nvcc_path,
        "no_global_cuda_mutation": True,
        "next_action": "Install cumm into the managed venv before attempting the CUDA resolver patch.",
    }
    if common_path is None:
        result["reason"] = "missing_cumm_common_py"
        return result
    if not toolkit_root or not include_dir or not nvcc_path:
        result["reason"] = "missing_selected_cuda_toolkit"
        result["next_action"] = "Resolve the selected CUDA toolkit metadata before attempting the managed cumm CUDA resolver patch."
        return result

    source = common_path.read_text(encoding="utf-8")
    patched_source, patch_status, patch_reason = _canonicalize_cumm_cuda_resolver(source)
    if patch_reason is not None:
        result["status"] = "failed"
        result["reason"] = patch_reason
        result["next_action"] = (
            "Inspect the installed cumm/common.py manually because the guarded CUDA resolver patch only supports the known upstream Linux resolver layout."
        )
        return result
    if patch_status == "already_patched":
        result["status"] = "already_patched"
        result["action"] = "noop"
        result["next_action"] = "None. guarded cumm CUDA resolver patch is already present."
        return result

    common_path.write_text(patched_source, encoding="utf-8")
    verified_source = common_path.read_text(encoding="utf-8")
    result["attempted"] = True
    result["action"] = "rewrite"
    _, verified_status, verified_reason = _canonicalize_cumm_cuda_resolver(verified_source)
    if verified_reason is not None or verified_status != "already_patched":
        result["status"] = "failed"
        result["reason"] = "cuda_resolver_patch_verification_failed"
        result["next_action"] = "Inspect the rewritten cumm/common.py because the guarded CUDA resolver patch could not be verified."
        return result

    result["status"] = "patched"
    result["next_action"] = "None. guarded cumm CUDA resolver patch is in place."
    return result


def _canonicalize_cumm_cuda_resolver(source: str) -> tuple[str, str, str | None]:
    if CUMM_CUDA_RESOLVER_PATCH_SENTINEL in source:
        return source, "already_patched", None
    if CUMM_CUDA_RESOLVER_LINUX_FRAGMENT not in source:
        return source, "failed", "unknown_cumm_cuda_resolver_source"
    return (
        source.replace(CUMM_CUDA_RESOLVER_LINUX_FRAGMENT, CUMM_CUDA_RESOLVER_PATCHED_LINUX_FRAGMENT, 1),
        "patched",
        None,
    )


def _canonicalize_cumm_dtype_header(
    *,
    name: str,
    relative_path: Path,
    legacy_fragments: tuple[GUARDED_FRAGMENT_SPEC, ...],
    source: str,
) -> tuple[str, str, str | None]:
    target_source = _local_target_cumm_dtype_header_source(
        name=name,
        relative_path=relative_path,
        legacy_fragments=legacy_fragments,
    )
    if source == target_source:
        return source, "already_patched", None

    supported_sources = _known_supported_cumm_dtype_header_sources(
        name=name,
        relative_path=relative_path,
        legacy_fragments=legacy_fragments,
    )
    if source in supported_sources:
        return target_source, "patched", None
    return source, "failed", "unknown_header_source"


def _read_cumm_header_snapshot(version: str, relative_path: Path) -> str:
    return (CUMM_HEADER_SNAPSHOT_ROOT / version / "include" / relative_path).read_text(encoding="utf-8")


def _local_target_cumm_dtype_header_source(
    *,
    name: str,
    relative_path: Path,
    legacy_fragments: tuple[GUARDED_FRAGMENT_SPEC, ...],
) -> str:
    canonical_source = _read_cumm_header_snapshot(CUMM_DTYPE_HEADER_CANONICAL_VERSION, relative_path)
    target_source = _compiler_safe_cumm_dtype_header_source(name=name, canonical_source=canonical_source)
    if target_source == canonical_source or name == "tf32":
        return target_source
    legacy_target_source = _legacy_local_target_cumm_dtype_header_source(
        canonical_source=canonical_source,
        name=name,
        legacy_fragments=legacy_fragments,
    )
    if target_source == legacy_target_source:
        raise RuntimeError(f"compiler-safe local target for cumm {name} unexpectedly matches the legacy target")
    return target_source


def _legacy_local_target_cumm_dtype_header_source(
    *,
    canonical_source: str,
    name: str,
    legacy_fragments: tuple[GUARDED_FRAGMENT_SPEC, ...],
) -> str:
    target_source, target_status, target_reason = _canonicalize_guarded_text_fragments(
        canonical_source,
        legacy_fragments,
    )
    if target_reason is not None:
        raise RuntimeError(f"failed to derive legacy local target for cumm {name}: {target_reason}")
    if target_status not in {"already_patched", "patched"}:
        raise RuntimeError(f"unexpected legacy local target status for cumm {name}: {target_status}")
    return target_source


def _compiler_safe_cumm_dtype_header_source(*, name: str, canonical_source: str) -> str:
    if name == "tf32":
        return _compiler_safe_tf32_header_source(canonical_source)
    if name == "float8":
        return _compiler_safe_float8_header_source(canonical_source)
    raise RuntimeError(f"unsupported cumm dtype header target: {name}")


def _compiler_safe_tf32_header_source(canonical_source: str) -> str:
    source = _latest_live_broken_tf32_header_source(canonical_source)
    source = _inject_cumm_host_math_macro_undef_block(
        source,
        anchor="#include <tensorview/core/all.h>\n",
        names=CUMM_HOST_MATH_MACRO_NAMES,
    )
    if source is None:
        raise RuntimeError("failed to inject tf32 host math macro undef block")
    source = _canonicalize_cumm_math_namespace_free_functions(
        source,
        scalar_type="tv::tfloat32_t",
        copysign_signature=(
            "static TV_HOST_DEVICE_INLINE\ntv::tfloat32_t copysign(tv::tfloat32_t const &a, tv::tfloat32_t const &b) {",
            "TV_HOST_DEVICE_INLINE\ntfloat32_t copysign(tfloat32_t const &a, tfloat32_t const &b) {",
            "TV_HOST_DEVICE_INLINE\ntv::tfloat32_t copysign(tv::tfloat32_t const &a, tv::tfloat32_t const &b) {",
            "static TV_HOST_DEVICE_INLINE\ntfloat32_t copysign(tfloat32_t const &a, tfloat32_t const &b) {",
        ),
    )
    if source is None:
        raise RuntimeError("failed to canonicalize tf32 namespace-scope math helpers")
    return source


def _latest_live_broken_tf32_header_source(canonical_source: str) -> str:
    source = _tf32_without_cuda_namespace_macros(canonical_source)
    source = _replace_exact_once(
        source,
        "namespace CUDA_NAMESPACE_STD {\n\n#if !defined(__CUDACC_RTC__)\n",
        "#if !defined(__CUDACC_RTC__)\nnamespace std {\n\n",
    )
    if source is None:
        raise RuntimeError("failed to rewrite tf32 standard namespace block start to host-only form")
    source = _replace_exact_once(
        source,
        "#endif\n\n///////////////////////////////////////////////////////////////////////////////////////////////////\n\n} // namespace std\n",
        "///////////////////////////////////////////////////////////////////////////////////////////////////\n\n} // namespace std\n#endif\n",
    )
    if source is None:
        raise RuntimeError("failed to rewrite tf32 standard namespace block end to host-only form")
    return source


def _tf32_without_cuda_namespace_macros(canonical_source: str) -> str:
    source = _replace_exact_once(
        canonical_source,
        "#if !defined(__CUDACC_RTC__)\n#include <cmath>\n#include <cstdint>\n#include <limits>\n#define CUDA_NAMESPACE_STD std\n#else \n#include \"fp_nvrtc.h\"\n#include <tensorview/core/nvrtc_std.h>\n#define CUDA_NAMESPACE_STD cuda::std\n#endif\n",
        "#if !defined(__CUDACC_RTC__)\n#include <cmath>\n#include <cstdint>\n#include <limits>\n#else \n#include \"fp_nvrtc.h\"\n#include <tensorview/core/nvrtc_std.h>\n#endif\n",
    )
    if source is None:
        raise RuntimeError("failed to remove tf32 CUDA namespace macro guards from canonical snapshot")
    return source


def _previous_local_target_tf32_header_source(canonical_source: str) -> str:
    source = _tf32_without_cuda_namespace_macros(canonical_source)
    source = _replace_exact_once(source, "namespace CUDA_NAMESPACE_STD {\n", "namespace std {\n")
    if source is None:
        raise RuntimeError("failed to derive previous tf32 target namespace form")
    return source


def _compiler_safe_float8_header_source(canonical_source: str) -> str:
    source = _latest_live_broken_float8_header_source(canonical_source)
    source = _inject_cumm_host_math_macro_undef_block(
        source,
        anchor="#include <tensorview/core/all.h>\n",
        names=("isfinite", "isnan", "isinf"),
    )
    if source is None:
        raise RuntimeError("failed to inject float8 host math macro undef block")
    replacements = (
        (
            "    static constexpr int FP8_NUM_EXPONENT_BITS = self_type::IS_E4M3 ? 4 : 5;\n",
            "    static constexpr int FP8_NUM_EXPONENT_BITS = (T == FloatEncoding::E4M3) ? 4 : 5;\n",
        ),
        (
            "    static constexpr int FP8_NUM_MANTISSA_BITS = self_type::IS_E4M3 ? 3 : 2;\n",
            "    static constexpr int FP8_NUM_MANTISSA_BITS = (T == FloatEncoding::E4M3) ? 3 : 2;\n",
        ),
        (
            "    static constexpr uint8_t  FP8_INFINITY_MASK = self_type::IS_E4M3 ? 0x78 : 0x7c;\n",
            "    static constexpr uint8_t  FP8_INFINITY_MASK = (T == FloatEncoding::E4M3) ? 0x78 : 0x7c;\n",
        ),
        (
            "    static constexpr int FP8_MAX_EXPONENT  = self_type::IS_E4M3 ?  7 :  15;\n",
            "    static constexpr int FP8_MAX_EXPONENT  = (T == FloatEncoding::E4M3) ?  7 :  15;\n",
        ),
        (
            "    static constexpr int FP8_MIN_EXPONENT  = self_type::IS_E4M3 ? -6 : -14;\n",
            "    static constexpr int FP8_MIN_EXPONENT  = (T == FloatEncoding::E4M3) ? -6 : -14;\n",
        ),
        (
            "    static constexpr int FP8_EXPONENT_BIAS = self_type::IS_E4M3 ?  7 :  15;\n",
            "    static constexpr int FP8_EXPONENT_BIAS = (T == FloatEncoding::E4M3) ?  7 :  15;\n",
        ),
        (
            "    static constexpr uint8_t FP8_MAX_FLT = (self_type::IS_E4M3 ? 0x7e : 0x7b);\n",
            "    static constexpr uint8_t FP8_MAX_FLT = ((T == FloatEncoding::E4M3) ? 0x7e : 0x7b);\n",
        ),
        (
            "    static constexpr uint8_t  FP8_EXPONENT_MASK = (1 << self_type::FP8_NUM_EXPONENT_BITS) - 1;\n",
            "    static constexpr uint8_t  FP8_EXPONENT_MASK = uint8_t((1u << ((T == FloatEncoding::E4M3) ? 4 : 5)) - 1u);\n",
        ),
        (
            "    static constexpr uint8_t  FP8_MANTISSA_MASK = (1 << self_type::FP8_NUM_MANTISSA_BITS) - 1;\n",
            "    static constexpr uint8_t  FP8_MANTISSA_MASK = uint8_t((1u << ((T == FloatEncoding::E4M3) ? 3 : 2)) - 1u);\n",
        ),
    )
    for old, new in replacements:
        source = _replace_all_required(source, old, new)
        if source is None:
            raise RuntimeError("failed to rewrite float8 fragile dependent-name constants to direct constexpr form")
    return source


def _latest_live_broken_float8_header_source(canonical_source: str) -> str:
    source = _replace_exact_once(
        canonical_source,
        "#if !defined(__CUDACC_RTC__)\n#include <cmath>\n#include <cstdint>\n#include <limits>\n#include  <cstring>\n#define CUDA_NAMESPACE_STD std\n#else \n#include \"fp_nvrtc.h\"\n#include <tensorview/core/nvrtc_std.h>\n#define CUDA_NAMESPACE_STD cuda::std\n#endif\n",
        "#if !defined(__CUDACC_RTC__)\n#include <cmath>\n#include <cstdint>\n#include <limits>\n#include  <cstring>\n#else \n#include \"fp_nvrtc.h\"\n#include <tensorview/core/nvrtc_std.h>\n#endif\n",
    )
    if source is None:
        raise RuntimeError("failed to remove float8 CUDA namespace macro guards from canonical snapshot")
    source = _rewrite_cumm_std_namespace_block_host_only(
        source,
        start_marker="namespace CUDA_NAMESPACE_STD {\n",
        end_marker="}  // namespace std\n",
    )
    if source is None:
        raise RuntimeError("failed to rewrite float8 standard namespace block to host-only form")
    replacements = (
        (
            "    static constexpr bool IS_E5M2 = (T == FloatEncoding::E5M2);\n",
            "    static constexpr bool IS_E5M2 = (T == FloatEncoding::E5M2);\n    using self_type = float8_base<T>;\n",
        ),
        (
            "    static constexpr int FP8_NUM_EXPONENT_BITS = IS_E4M3 ? 4 : 5;\n",
            "    static constexpr int FP8_NUM_EXPONENT_BITS = self_type::IS_E4M3 ? 4 : 5;\n",
        ),
        (
            "    static constexpr int FP8_NUM_MANTISSA_BITS = IS_E4M3 ? 3 : 2;\n",
            "    static constexpr int FP8_NUM_MANTISSA_BITS = self_type::IS_E4M3 ? 3 : 2;\n",
        ),
        (
            "    static constexpr uint8_t  FP8_INFINITY_MASK = IS_E4M3 ? 0x78 : 0x7c;\n",
            "    static constexpr uint8_t  FP8_INFINITY_MASK = self_type::IS_E4M3 ? 0x78 : 0x7c;\n",
        ),
        (
            "    static constexpr int FP8_MAX_EXPONENT  = IS_E4M3 ?  7 :  15;\n",
            "    static constexpr int FP8_MAX_EXPONENT  = self_type::IS_E4M3 ?  7 :  15;\n",
        ),
        (
            "    static constexpr int FP8_MIN_EXPONENT  = IS_E4M3 ? -6 : -14;\n",
            "    static constexpr int FP8_MIN_EXPONENT  = self_type::IS_E4M3 ? -6 : -14;\n",
        ),
        (
            "    static constexpr int FP8_EXPONENT_BIAS = IS_E4M3 ?  7 :  15;\n",
            "    static constexpr int FP8_EXPONENT_BIAS = self_type::IS_E4M3 ?  7 :  15;\n",
        ),
        (
            "    static constexpr uint8_t FP8_MAX_FLT = (IS_E4M3 ? 0x7e : 0x7b);\n",
            "    static constexpr uint8_t FP8_MAX_FLT = (self_type::IS_E4M3 ? 0x7e : 0x7b);\n",
        ),
        (
            "    static constexpr uint8_t  FP8_EXPONENT_MASK = (1 << FP8_NUM_EXPONENT_BITS) - 1;\n",
            "    static constexpr uint8_t  FP8_EXPONENT_MASK = (1 << self_type::FP8_NUM_EXPONENT_BITS) - 1;\n",
        ),
        (
            "    static constexpr uint8_t  FP8_MANTISSA_MASK = (1 << FP8_NUM_MANTISSA_BITS) - 1;\n",
            "    static constexpr uint8_t  FP8_MANTISSA_MASK = (1 << self_type::FP8_NUM_MANTISSA_BITS) - 1;\n",
        ),
        (
            "        int8_t exp = uint8_t(((s >> FP32_NUM_MANTISSA_BITS) & 0xff) - FP32_EXPONENT_BIAS);\n",
            "        int8_t exp = uint8_t(((s >> self_type::FP32_NUM_MANTISSA_BITS) & 0xff) - self_type::FP32_EXPONENT_BIAS);\n",
        ),
        (
            "        if (isnan(flt)) {\n",
            "        if (self_type::isnan(flt)) {\n",
        ),
        (
            "        if (isinf(flt)) {\n",
            "        if (self_type::isinf(flt)) {\n",
        ),
        (
            "            return sign | FP8_MAX_FLT;\n",
            "            return sign | self_type::FP8_MAX_FLT;\n",
        ),
        (
            "            return (sign | FP8_MAX_FLT);\n",
            "            return (sign | self_type::FP8_MAX_FLT);\n",
        ),
        (
            "        if ( (exp >= FP8_MIN_EXPONENT) && (exp <= FP8_MAX_EXPONENT) ) {\n",
            "        if ( (exp >= self_type::FP8_MIN_EXPONENT) && (exp <= self_type::FP8_MAX_EXPONENT) ) {\n",
        ),
        (
            "            exp = uint8_t(exp + uint8_t(FP8_EXPONENT_BIAS));\n",
            "            exp = uint8_t(exp + uint8_t(self_type::FP8_EXPONENT_BIAS));\n",
        ),
        (
            "            u = uint8_t(((exp & FP8_EXPONENT_MASK) << FP8_NUM_MANTISSA_BITS));\n",
            "            u = uint8_t(((exp & self_type::FP8_EXPONENT_MASK) << self_type::FP8_NUM_MANTISSA_BITS));\n",
        ),
        (
            "            u = uint8_t(u | (mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)));\n",
            "            u = uint8_t(u | (mantissa >> (self_type::FP32_NUM_MANTISSA_BITS - self_type::FP8_NUM_MANTISSA_BITS)));\n",
        ),
        (
            "        } else if(exp < FP8_MIN_EXPONENT) {\n",
            "        } else if(exp < self_type::FP8_MIN_EXPONENT) {\n",
        ),
        (
            "            int rshift = (FP8_MIN_EXPONENT - exp);\n",
            "            int rshift = (self_type::FP8_MIN_EXPONENT - exp);\n",
        ),
        (
            "            if (rshift < FP32_NUM_BITS) {\n",
            "            if (rshift < self_type::FP32_NUM_BITS) {\n",
        ),
        (
            "                mantissa |= (1 << FP32_NUM_MANTISSA_BITS);\n",
            "                mantissa |= (1 << self_type::FP32_NUM_MANTISSA_BITS);\n",
        ),
        (
            "                u = (uint8_t(mantissa >> (FP32_NUM_MANTISSA_BITS- FP8_NUM_MANTISSA_BITS)) & FP8_MANTISSA_MASK);\n",
            "                u = (uint8_t(mantissa >> (self_type::FP32_NUM_MANTISSA_BITS- self_type::FP8_NUM_MANTISSA_BITS)) & self_type::FP8_MANTISSA_MASK);\n",
        ),
        (
            "            if( exp == (FP8_MAX_EXPONENT + 1) ) {\n",
            "            if( exp == (self_type::FP8_MAX_EXPONENT + 1) ) {\n",
        ),
        (
            "                uint8_t mantissa_tmp = uint8_t(mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));\n",
            "                uint8_t mantissa_tmp = uint8_t(mantissa >> (self_type::FP32_NUM_MANTISSA_BITS - self_type::FP8_NUM_MANTISSA_BITS));\n",
        ),
        (
            "                if( mantissa_tmp < FP8_MANTISSA_MASK) {\n",
            "                if( mantissa_tmp < self_type::FP8_MANTISSA_MASK) {\n",
        ),
        (
            "                    u = uint8_t(exp << FP8_NUM_MANTISSA_BITS) | mantissa_tmp;\n",
            "                    u = uint8_t(exp << self_type::FP8_NUM_MANTISSA_BITS) | mantissa_tmp;\n",
        ),
        (
            "                    may_be_nan =  (mantissa_tmp == (FP8_MANTISSA_MASK-1));\n",
            "                    may_be_nan =  (mantissa_tmp == (self_type::FP8_MANTISSA_MASK-1));\n",
        ),
        (
            "        int NUM_BITS_SHIFT = FP32_NUM_MANTISSA_BITS - (FP8_NUM_MANTISSA_BITS + 1);\n",
            "        int NUM_BITS_SHIFT = self_type::FP32_NUM_MANTISSA_BITS - (self_type::FP8_NUM_MANTISSA_BITS + 1);\n",
        ),
        (
            "        if (u > FP8_MAX_FLT) {\n",
            "        if (u > self_type::FP8_MAX_FLT) {\n",
        ),
        (
            "            u = (sign | FP8_MAX_FLT);\n",
            "            u = (sign | self_type::FP8_MAX_FLT);\n",
        ),
        (
            "        int sign = (f8 >> (FP8_NUM_BITS - 1)) & 1;\n",
            "        int sign = (f8 >> (self_type::FP8_NUM_BITS - 1)) & 1;\n",
        ),
        (
            "        int exp = (f8 >> FP8_NUM_MANTISSA_BITS) & FP8_EXPONENT_MASK;\n",
            "        int exp = (f8 >> self_type::FP8_NUM_MANTISSA_BITS) & self_type::FP8_EXPONENT_MASK;\n",
        ),
        (
            "        int mantissa = f8 & FP8_MANTISSA_MASK;\n",
            "        int mantissa = f8 & self_type::FP8_MANTISSA_MASK;\n",
        ),
        (
            "        unsigned f = (sign << (FP32_NUM_BITS-1));\n",
            "        unsigned f = (sign << (self_type::FP32_NUM_BITS-1));\n",
        ),
        (
            "        if (IS_E4M3 && exp == 15 && mantissa == 0x7) {\n",
            "        if (self_type::IS_E4M3 && exp == 15 && mantissa == 0x7) {\n",
        ),
        (
            "        else if (exp > 0 && (IS_E4M3 || exp < (FP8_MAX_EXPONENT + FP8_EXPONENT_BIAS + 1))) {\n",
            "        else if (exp > 0 && (self_type::IS_E4M3 || exp < (self_type::FP8_MAX_EXPONENT + self_type::FP8_EXPONENT_BIAS + 1))) {\n",
        ),
        (
            "            exp += (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS);\n",
            "            exp += (self_type::FP32_EXPONENT_BIAS - self_type::FP8_EXPONENT_BIAS);\n",
        ),
        (
            "                (exp << FP32_NUM_MANTISSA_BITS) |\n",
            "                (exp << self_type::FP32_NUM_MANTISSA_BITS) |\n",
        ),
        (
            "                (mantissa << (FP32_NUM_MANTISSA_BITS-FP8_NUM_MANTISSA_BITS));\n",
            "                (mantissa << (self_type::FP32_NUM_MANTISSA_BITS-self_type::FP8_NUM_MANTISSA_BITS));\n",
        ),
        (
            "                exp += (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS) + 1;\n",
            "                exp += (self_type::FP32_EXPONENT_BIAS - self_type::FP8_EXPONENT_BIAS) + 1;\n",
        ),
        (
            "                while ((mantissa & (1 << FP8_NUM_MANTISSA_BITS)) == 0) {\n",
            "                while ((mantissa & (1 << self_type::FP8_NUM_MANTISSA_BITS)) == 0) {\n",
        ),
        (
            "                mantissa &= FP8_MANTISSA_MASK;\n",
            "                mantissa &= self_type::FP8_MANTISSA_MASK;\n",
        ),
        (
            "        return int((storage >> FP8_NUM_MANTISSA_BITS) & Base::FP8_EXPONENT_MASK);\n",
            "        return int((storage >> Base::FP8_NUM_MANTISSA_BITS) & Base::FP8_EXPONENT_MASK);\n",
        ),
    )
    for old, new in replacements:
        source = _replace_all_required(source, old, new)
        if source is None:
            raise RuntimeError("failed to rewrite float8 compiler-safe constants, helper calls, or namespace form")
    return source


def _inject_cumm_host_math_macro_undef_block(source: str, *, anchor: str, names: tuple[str, ...]) -> str | None:
    block_lines: list[str] = []
    for name in names:
        block_lines.extend((f"#ifdef {name}\n", f"#undef {name}\n", "#endif\n"))
    block = "".join(block_lines)
    if block in source:
        return source
    return _replace_exact_once(source, anchor, f"{anchor}\n{block}")


def _canonicalize_cumm_half_convert_member(source: str) -> str | None:
    member_signatures = (
        (
            """  TV_HOST_DEVICE_INLINE
  static half_t convert(float const &flt) {
""",
            (
                """#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)
  // Avoid inlining in device code if no hardware support
  static __device__ __noinline__ half_t convert(float const &flt) {
#else
  static TV_HOST_DEVICE_INLINE half_t convert(float const &flt) {
#endif
""",
                """#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)
  // Avoid inlining in device code if no hardware support
  __device__ __noinline__
#else
  TV_HOST_DEVICE_INLINE
#endif
      static half_t
      convert(float const &flt) {
""",
                """#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)
  // Avoid inlining in device code if no hardware support
  __device__ __noinline__
#else
  TV_HOST_DEVICE_INLINE
#endif
  static half_t convert(float const &flt) {
""",
            ),
        ),
        (
            """  TV_HOST_DEVICE_INLINE
  static float convert(half_t const &x) {
""",
            (
                """#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)
  // Avoid inlining in device code if no hardware support
  static __device__ __noinline__ float convert(half_t const &x) {
#else
  static TV_HOST_DEVICE_INLINE float convert(half_t const &x) {
#endif
""",
                """#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)
  // Avoid inlining in device code if no hardware support
  __device__ __noinline__
#else
  TV_HOST_DEVICE_INLINE
#endif
      static float
      convert(half_t const &x) {
""",
                """#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)
  // Avoid inlining in device code if no hardware support
  __device__ __noinline__
#else
  TV_HOST_DEVICE_INLINE
#endif
  static float convert(half_t const &x) {
""",
            ),
        ),
    )
    target_source = source
    for target_signature, candidate_signatures in member_signatures:
        if target_signature in target_source:
            continue
        for candidate in candidate_signatures:
            if candidate in target_source:
                target_source = target_source.replace(candidate, target_signature, 1)
                break
        else:
            return None
    return target_source


def _split_cumm_math_namespace_free_function_region(source: str, *, struct_name: str) -> tuple[str, str, str] | None:
    struct_marker = f"struct alignas(2) {struct_name} {{"
    struct_start = source.find(struct_marker)
    if struct_start == -1:
        return None
    struct_end = source.find("\n};\n", struct_start)
    if struct_end == -1:
        return None
    namespace_tv_end = source.find("\n} // namespace tv\n", struct_end)
    namespace_std_start = source.find("\nnamespace std {\n", struct_end)
    if namespace_tv_end == -1 or namespace_std_start == -1 or namespace_tv_end >= namespace_std_start:
        return None
    helper_start = struct_end + len("\n};\n")
    return source[:helper_start], source[helper_start:namespace_tv_end], source[namespace_tv_end:]


def _canonicalize_cumm_math_namespace_free_function_region(
    source: str,
    *,
    struct_name: str,
    scalar_type: str,
    copysign_signature: tuple[str, str, str, str],
    drop_signbit: bool = False,
) -> str | None:
    split = _split_cumm_math_namespace_free_function_region(source, struct_name=struct_name)
    if split is None:
        return None
    prefix, helper_region, suffix = split
    target_source = helper_region
    signbit_static_signature = f"static TV_HOST_DEVICE_INLINE\nbool signbit({scalar_type} const &h) {{"
    signbit_plain_signature = f"TV_HOST_DEVICE_INLINE\nbool signbit({scalar_type} const &h) {{"
    if drop_signbit:
        if signbit_static_signature in target_source:
            target_source = target_source.replace(f"{signbit_static_signature} return h.signbit(); }}\n\n", "", 1)
        elif signbit_plain_signature in target_source:
            target_source = target_source.replace(f"{signbit_plain_signature} return h.signbit(); }}\n\n", "", 1)
        elif "signbit(" in target_source:
            return None
    namespace_signatures = (
        (f"static TV_HOST_DEVICE_INLINE\nbool isnan({scalar_type} const &h) {{", f"TV_HOST_DEVICE_INLINE\nbool isnan({scalar_type} const &h) {{"),
        (f"static TV_HOST_DEVICE_INLINE\nbool isfinite({scalar_type} const &h) {{", f"TV_HOST_DEVICE_INLINE\nbool isfinite({scalar_type} const &h) {{"),
        (f"static TV_HOST_DEVICE_INLINE\nbool isinf({scalar_type} const &h) {{", f"TV_HOST_DEVICE_INLINE\nbool isinf({scalar_type} const &h) {{"),
        (f"static TV_HOST_DEVICE_INLINE\nbool isnormal({scalar_type} const &h) {{", f"TV_HOST_DEVICE_INLINE\nbool isnormal({scalar_type} const &h) {{"),
        (f"static TV_HOST_DEVICE_INLINE\nint fpclassify({scalar_type} const &h) {{", f"TV_HOST_DEVICE_INLINE\nint fpclassify({scalar_type} const &h) {{"),
        (f"static TV_HOST_DEVICE_INLINE\n{scalar_type} sqrt({scalar_type} const &h) {{", f"TV_HOST_DEVICE_INLINE\n{scalar_type} sqrt({scalar_type} const &h) {{"),
    )
    if not drop_signbit:
        namespace_signatures = ((signbit_static_signature, signbit_plain_signature), *namespace_signatures)
    for static_signature, plain_signature in namespace_signatures:
        if static_signature in target_source:
            continue
        if plain_signature in target_source:
            target_source = target_source.replace(plain_signature, static_signature)
            continue
        return None
    target_signature, *candidate_signatures = copysign_signature
    if target_signature in target_source:
        return f"{prefix}{target_source}{suffix}"
    for candidate in candidate_signatures:
        if candidate in target_source:
            return f"{prefix}{target_source.replace(candidate, target_signature, 1)}{suffix}"
    return None


def _canonicalize_cumm_math_namespace_free_functions(
    source: str,
    *,
    scalar_type: str,
    copysign_signature: tuple[str, str, str, str],
) -> str | None:
    target_source = source
    namespace_signatures = (
        (f"static TV_HOST_DEVICE_INLINE\nbool signbit({scalar_type} const &h) {{", f"TV_HOST_DEVICE_INLINE\nbool signbit({scalar_type} const &h) {{"),
        (f"static TV_HOST_DEVICE_INLINE\nbool isnan({scalar_type} const &h) {{", f"TV_HOST_DEVICE_INLINE\nbool isnan({scalar_type} const &h) {{"),
        (f"static TV_HOST_DEVICE_INLINE\nbool isfinite({scalar_type} const &h) {{", f"TV_HOST_DEVICE_INLINE\nbool isfinite({scalar_type} const &h) {{"),
        (f"static TV_HOST_DEVICE_INLINE\nbool isinf({scalar_type} const &h) {{", f"TV_HOST_DEVICE_INLINE\nbool isinf({scalar_type} const &h) {{"),
        (f"static TV_HOST_DEVICE_INLINE\nbool isnormal({scalar_type} const &h) {{", f"TV_HOST_DEVICE_INLINE\nbool isnormal({scalar_type} const &h) {{"),
        (f"static TV_HOST_DEVICE_INLINE\nint fpclassify({scalar_type} const &h) {{", f"TV_HOST_DEVICE_INLINE\nint fpclassify({scalar_type} const &h) {{"),
        (f"static TV_HOST_DEVICE_INLINE\n{scalar_type} sqrt({scalar_type} const &h) {{", f"TV_HOST_DEVICE_INLINE\n{scalar_type} sqrt({scalar_type} const &h) {{"),
    )
    for static_signature, plain_signature in namespace_signatures:
        if static_signature in target_source:
            continue
        if plain_signature in target_source:
            target_source = target_source.replace(plain_signature, static_signature)
            continue
        return None
    target_signature, *candidate_signatures = copysign_signature
    if target_signature in target_source:
        return target_source
    for candidate in candidate_signatures:
        if candidate in target_source:
            return target_source.replace(candidate, target_signature)
    return None


def _previous_local_target_float8_header_source(canonical_source: str) -> str:
    source = _replace_exact_once(
        canonical_source,
        "#if !defined(__CUDACC_RTC__)\n#include <cmath>\n#include <cstdint>\n#include <limits>\n#include  <cstring>\n#define CUDA_NAMESPACE_STD std\n#else \n#include \"fp_nvrtc.h\"\n#include <tensorview/core/nvrtc_std.h>\n#define CUDA_NAMESPACE_STD cuda::std\n#endif\n",
        "#if !defined(__CUDACC_RTC__)\n#include <cmath>\n#include <cstdint>\n#include <limits>\n#include  <cstring>\n#else \n#include \"fp_nvrtc.h\"\n#include <tensorview/core/nvrtc_std.h>\n#endif\n",
    )
    if source is None:
        raise RuntimeError("failed to derive previous float8 target from canonical snapshot")
    source = _rewrite_cumm_std_namespace_block(
        source,
        start_marker="namespace CUDA_NAMESPACE_STD {\n",
        end_marker="}  // namespace std\n",
    )
    if source is None:
        raise RuntimeError("failed to derive previous float8 target namespace form")
    return source


def _rewrite_cumm_std_namespace_block(source: str, *, start_marker: str, end_marker: str) -> str | None:
    rewritten = _replace_exact_once(
        source,
        start_marker,
        "#if !defined(__CUDACC_RTC__)\nnamespace std {\n#else\nnamespace cuda {\nnamespace std {\n#endif\n",
    )
    if rewritten is None:
        return None
    return _replace_exact_once(
        rewritten,
        end_marker,
        "#if !defined(__CUDACC_RTC__)\n} // namespace std\n#else\n} // namespace std\n} // namespace cuda\n#endif\n",
    )


def _rewrite_cumm_std_namespace_block_host_only(source: str, *, start_marker: str, end_marker: str) -> str | None:
    rewritten = _replace_exact_once(
        source,
        start_marker,
        "#if !defined(__CUDACC_RTC__)\nnamespace std {\n",
    )
    if rewritten is None:
        return None
    return _replace_exact_once(
        rewritten,
        end_marker,
        "} // namespace std\n#endif\n",
    )


def _known_supported_cumm_dtype_header_sources(
    *,
    name: str,
    relative_path: Path,
    legacy_fragments: tuple[GUARDED_FRAGMENT_SPEC, ...],
) -> tuple[str, ...]:
    legacy_source = _read_cumm_header_snapshot(CUMM_DTYPE_HEADER_LEGACY_VERSION, relative_path)
    canonical_source = _read_cumm_header_snapshot(CUMM_DTYPE_HEADER_CANONICAL_VERSION, relative_path)
    candidates = [legacy_source, canonical_source]
    legacy_target_source = _legacy_local_target_cumm_dtype_header_source(
        canonical_source=canonical_source,
        name=name,
        legacy_fragments=legacy_fragments,
    )
    candidates.append(legacy_target_source)
    fragment_variant, fragment_status, fragment_reason = _canonicalize_guarded_text_fragments(legacy_source, legacy_fragments)
    if fragment_status == "patched" and fragment_reason is None:
        candidates.append(fragment_variant)

    if name == "tf32":
        candidates.append(_latest_live_broken_tf32_header_source(canonical_source))
        candidates.append(_previous_local_target_tf32_header_source(canonical_source))
        candidates.extend(_legacy_tf32_malformed_variants(legacy_source))
    elif name == "float8":
        candidates.append(_latest_live_broken_float8_header_source(canonical_source))
        candidates.append(_previous_local_target_float8_header_source(canonical_source))
        candidates.extend(_legacy_float8_malformed_variants(legacy_source))
        approximate_variant = _legacy_float8_approximate_variant(canonical_source)
        if approximate_variant is not None:
            candidates.append(approximate_variant)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate)
    return tuple(deduped)


def _legacy_tf32_malformed_variants(source: str) -> tuple[str, ...]:
    variants: list[str] = []
    with_std_define = _replace_exact_once(source, "#include <limits>\n", "#include <limits>\n#define CUDA_NAMESPACE_STD std\n")
    if with_std_define is not None:
        variants.append(with_std_define)
        with_namespace_macro = _replace_exact_once(
            with_std_define,
            "namespace std {\n",
            "namespace CUDA_NAMESPACE_STD {\n",
        )
        if with_namespace_macro is not None:
            variants.append(with_namespace_macro)
    return tuple(variants)


def _legacy_float8_malformed_variants(source: str) -> tuple[str, ...]:
    variants: list[str] = []
    with_std_define = _replace_exact_once(source, "#include  <cstring>\n", "#include  <cstring>\n#define CUDA_NAMESPACE_STD std\n")
    if with_std_define is not None:
        variants.append(with_std_define)

    replacements = (
        (
            "    static constexpr uint8_t  FP8_INFINITY_MASK = IS_E4M3 ? 0x78 : 0x7c;\n",
            "    static constexpr uint8_t  FP8_INFINITY_MASK = (T == FloatEncoding::E4M3) ? 0x78 : 0x7c;\n",
        ),
        (
            "    static constexpr int FP8_EXPONENT_BIAS = IS_E4M3 ?  7 :  15;\n",
            "    static constexpr int FP8_EXPONENT_BIAS = (T == FloatEncoding::E4M3) ?  7 :  15;\n",
        ),
        (
            "    static constexpr uint8_t FP8_MAX_FLT = (IS_E4M3 ? 0x7e : 0x7b);\n",
            "    static constexpr uint8_t FP8_MAX_FLT = ((IS_E4M3) ? 0x7e : 0x7b);\n",
        ),
    )
    malformed = with_std_define or source
    for old, new in replacements:
        replaced = _replace_exact_once(malformed, old, new)
        if replaced is None:
            malformed = ""
            break
        malformed = replaced
    if malformed:
        variants.append(malformed)
    return tuple(variants)


def _legacy_float8_approximate_variant(source: str) -> str | None:
    replacements = (
        (
            "    static constexpr int FP8_NUM_EXPONENT_BITS = IS_E4M3 ? 4 : 5;\n",
            "    static constexpr int FP8_NUM_EXPONENT_BITS = (T == FloatEncoding::E4M3) ? 4 : 5;\n",
        ),
        (
            "    static constexpr int FP8_NUM_MANTISSA_BITS = IS_E4M3 ? 3 : 2;\n",
            "    static constexpr int FP8_NUM_MANTISSA_BITS = (T == FloatEncoding::E4M3) ? 3 : 2;\n",
        ),
        (
            "    static constexpr uint8_t  FP8_INFINITY_MASK = IS_E4M3 ? 0x78 : 0x7c;\n",
            "    static constexpr uint8_t  FP8_INFINITY_MASK = (T == FloatEncoding::E4M3) ? 0x78 : 0x7c;\n",
        ),
        (
            "    static constexpr int FP8_MAX_EXPONENT  = IS_E4M3 ?  7 :  15;\n",
            "    static constexpr int FP8_MAX_EXPONENT  = (T == FloatEncoding::E4M3) ?  7 :  15;\n",
        ),
        (
            "    static constexpr int FP8_MIN_EXPONENT  = IS_E4M3 ? -6 : -14;\n",
            "    static constexpr int FP8_MIN_EXPONENT  = (T == FloatEncoding::E4M3) ? -6 : -14;\n",
        ),
        (
            "    static constexpr int FP8_EXPONENT_BIAS = IS_E4M3 ?  7 :  15;\n",
            "    static constexpr int FP8_EXPONENT_BIAS = (T == FloatEncoding::E4M3) ?  7 :  15;\n",
        ),
        (
            "    static constexpr uint8_t FP8_MAX_FLT = (IS_E4M3 ? 0x7e : 0x7b);\n",
            "    static constexpr uint8_t FP8_MAX_FLT = ((T == FloatEncoding::E4M3) ? 0x7e : 0x7b);\n",
        ),
    )
    approximate = source
    for old, new in replacements:
        approximate = _replace_exact_once(approximate, old, new)
        if approximate is None:
            return None
    return approximate


def _replace_exact_once(source: str, old: str, new: str) -> str | None:
    if source.count(old) != 1:
        return None
    return source.replace(old, new, 1)


def _replace_all_required(source: str, old: str, new: str) -> str | None:
    if source.count(old) < 1:
        return None
    return source.replace(old, new)


def _canonicalize_guarded_text_fragments(
    source: str,
    fragments: tuple[GUARDED_FRAGMENT_SPEC, ...],
) -> tuple[str, str, str | None]:
    patched = source
    patched_any = False
    for canonical, variants in fragments:
        canonical_count = patched.count(canonical)
        if canonical_count > 1:
            return source, "failed", "ambiguous_fragment_count"

        if canonical_count == 1:
            residual = patched.replace(canonical, "", 1)
            if any(residual.count(variant) > 0 for variant in variants):
                return source, "failed", "mixed_fragment_state"
            continue

        variant_counts = {variant: patched.count(variant) for variant in variants}
        matched_variants = [variant for variant, count in variant_counts.items() if count == 1]
        duplicate_variants = [variant for variant, count in variant_counts.items() if count > 1]
        if duplicate_variants:
            return source, "failed", "ambiguous_fragment_count"
        if len(matched_variants) != 1:
            if len(matched_variants) > 1:
                return source, "failed", "ambiguous_fragment_count"
            return source, "failed", "missing_expected_fragment"
        patched = patched.replace(matched_variants[0], canonical, 1)
        patched_any = True
    return patched, "patched" if patched_any else "already_patched", None


def install_native_runtime_step(*, venv_dir: Path, step: NativeInstallStep) -> dict[str, Any]:
    if step.status != "planned" or not step.pip_args:
        return {
            "package": step.package,
            "status": "blocked",
            "strategy": step.strategy,
            "reason": step.reason or "No runnable install step is available.",
            "attempted": False,
        }
    extension_dir = venv_dir.parent
    preservation = _native_build_preservation(step.package, extension_dir)
    command = [str(venv_python(venv_dir)), "-m", "pip", *step.pip_args]
    env = os.environ.copy()
    env.update(step.env or {})
    env.update(preservation["env"])
    result = subprocess.run(command, check=False, capture_output=True, text=True, env=env, cwd=str(extension_dir))
    _write_native_build_log(preservation["stdout_log_path"], result.stdout)
    _write_native_build_log(preservation["stderr_log_path"], result.stderr)
    metadata = {
        "command": command,
        "cwd": str(extension_dir),
        "artifact_root": preservation["root"],
        "artifact_tmp_dir": preservation["temp_dir"],
        "build_tracker_dir": preservation["build_tracker_dir"],
        "stdout_log_path": preservation["stdout_log_path"],
        "stderr_log_path": preservation["stderr_log_path"],
    }
    if result.returncode != 0:
        stdout_tail = _tail_lines(result.stdout)
        stderr_tail = _tail_lines(result.stderr)
        return {
            "package": step.package,
            "status": "failed",
            "strategy": step.strategy,
            "requirement": step.requirement,
            "pip_args": list(step.pip_args),
            "env": dict(step.env or {}),
            "attempted": True,
            "returncode": result.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "build_error_lines": _build_error_lines(result.stdout, result.stderr),
            **metadata,
        }
    return {
        "package": step.package,
        "status": "installed",
        "strategy": step.strategy,
        "requirement": step.requirement,
        "pip_args": list(step.pip_args),
        "env": dict(step.env or {}),
        "attempted": True,
        **metadata,
    }


def run_native_runtime_phase(
    *,
    venv_dir: Path,
    managed_python: Path,
    system_name: str,
    machine: str,
    py_tag: str,
    gpu_sm: int,
    cuda_version: int,
    native_mode: str,
    installer: Any = install_native_runtime_step,
    cumm_header_repairer: Any = repair_cumm_tensorview_headers,
    cumm_core_validator: Any = probe_cumm_core_cc,
    cumm_subint_header_patcher: Any = patch_cumm_subint_header,
    cumm_half_header_patcher: Any = patch_cumm_half_header,
    cumm_bfloat16_header_patcher: Any = patch_cumm_bfloat16_header,
    cumm_dtype_header_patcher: Any = patch_cumm_dtype_headers,
    cumm_cuda_resolver_patcher: Any = patch_cumm_cuda_resolver,
    spconv_core_validator: Any = probe_spconv_core_cc,
) -> dict[str, Any]:
    plan = build_native_runtime_plan(
        system_name=system_name,
        machine=machine,
        py_tag=py_tag,
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
        native_mode=native_mode,
        venv_dir=venv_dir,
    )
    resolved_mode = plan.native_mode
    if resolved_mode == NATIVE_MODE_SKIP:
        return {
            "mode": resolved_mode,
            "ready": False,
            "status": "skipped",
            "attempted_install": False,
            "host": plan.host.to_dict(),
            "plan": plan.to_dict(),
            "probe": None,
            "install_results": [],
            "message": "Native runtime phase was explicitly skipped.",
            "next_action": "Use native_mode=auto to probe or native_mode=install for guarded repair attempts.",
        }

    probe = probe_native_runtime(managed_python=managed_python, plan=plan)
    install_results: list[dict[str, Any]] = []
    attempted_install = False
    final_probe = probe
    status = probe.status
    message = probe.message
    next_action = probe.next_action

    auto_prebuilt_install = (
        resolved_mode == NATIVE_MODE_AUTO
        and plan.automatic_install_supported
        and plan.install_strategy == "windows-amd64-prebuilt-wheels"
        and not probe.ready
    )

    if (resolved_mode == NATIVE_MODE_INSTALL or auto_prebuilt_install) and not probe.ready:
        if not plan.automatic_install_supported:
            status = "blocked"
            message = plan.message
            next_action = plan.next_action
        else:
            attempted_install = True
            for step in plan.install_steps:
                try:
                    result = installer(venv_dir=venv_dir, step=step)
                except Exception as exc:
                    result = {
                        "package": step.package,
                        "status": "failed",
                        "strategy": step.strategy,
                        "requirement": step.requirement,
                        "pip_args": list(step.pip_args),
                        "env": dict(step.env or {}),
                        "attempted": True,
                        "reason": str(exc),
                        "error_type": type(exc).__name__,
                    }
                install_results.append(result)
                if result.get("status") != "installed":
                    status = "degraded"
                    message = (
                        "Native runtime source-build repair failed; inspect install_results for the failing step and logs."
                    )
                    next_action = (
                        "Fix the failing native build prerequisite or dependency step, then retry native_mode=install explicitly."
                    )
                    break
                if step.package == "cumm" and plan.install_strategy == "linux-arm64-source-build":
                    repair_result = cumm_header_repairer(venv_dir=venv_dir, extension_dir=venv_dir.parent)
                    install_results.append(repair_result)
                    if repair_result.get("status") not in {"already_present", "repaired"}:
                        status = "degraded"
                        message = (
                            "Native runtime source-build repair failed while restoring cumm tensorview headers; inspect install_results for details."
                        )
                        next_action = repair_result.get(
                            "next_action",
                            "Provide the missing cumm tensorview headers, then retry native_mode=install explicitly.",
                        )
                        break
                    validation_result = cumm_core_validator(managed_python=managed_python)
                    install_results.append(validation_result)
                    if not validation_result.get("ready", False):
                        status = "degraded"
                        message = (
                            "Native runtime source-build repair stopped because cumm.core_cc is not importable; spconv install was skipped."
                        )
                        next_action = validation_result.get(
                            "next_action",
                            "Repair the cumm native build so cumm.core_cc imports cleanly before retrying native_mode=install.",
                        )
                        break
                    patch_result = cumm_subint_header_patcher(venv_dir=venv_dir)
                    install_results.append(patch_result)
                    if patch_result.get("status") not in {"patched", "already_patched"}:
                        status = "degraded"
                        message = (
                            "Native runtime source-build repair stopped because the guarded cumm subint header patch failed; spconv install was skipped."
                        )
                        next_action = patch_result.get(
                            "next_action",
                            "Inspect the managed cumm subint header and retry native_mode=install explicitly before attempting spconv.",
                        )
                        break
                    half_patch_result = cumm_half_header_patcher(venv_dir=venv_dir)
                    install_results.append(half_patch_result)
                    if half_patch_result.get("status") not in {"patched", "already_patched"}:
                        status = "degraded"
                        message = (
                            "Native runtime source-build repair stopped because the guarded cumm half header patch failed; spconv install was skipped."
                        )
                        next_action = half_patch_result.get(
                            "next_action",
                            "Inspect the managed cumm half header and retry native_mode=install explicitly before attempting spconv.",
                        )
                        break
                    bfloat16_patch_result = cumm_bfloat16_header_patcher(venv_dir=venv_dir)
                    install_results.append(bfloat16_patch_result)
                    if bfloat16_patch_result.get("status") not in {"patched", "already_patched"}:
                        status = "degraded"
                        message = (
                            "Native runtime source-build repair stopped because the guarded cumm bfloat16 header patch failed; spconv install was skipped."
                        )
                        next_action = bfloat16_patch_result.get(
                            "next_action",
                            "Inspect the managed cumm bfloat16 header and retry native_mode=install explicitly before attempting spconv.",
                        )
                        break
                    dtype_patch_result = cumm_dtype_header_patcher(venv_dir=venv_dir)
                    install_results.append(dtype_patch_result)
                    if dtype_patch_result.get("status") not in {"patched", "already_patched"}:
                        status = "degraded"
                        message = (
                            "Native runtime source-build repair stopped because the guarded cumm dtype header patch failed; spconv install was skipped."
                        )
                        next_action = dtype_patch_result.get(
                            "next_action",
                            "Inspect the managed cumm tf32/float8 headers and retry native_mode=install explicitly before attempting spconv.",
                        )
                        break
                    cuda_resolver_patch_result = cumm_cuda_resolver_patcher(
                        venv_dir=venv_dir,
                        toolkit_root=plan.toolkit_root,
                        include_dir=plan.toolkit_include_dir,
                        nvcc_path=plan.toolkit_nvcc,
                    )
                    install_results.append(cuda_resolver_patch_result)
                    if cuda_resolver_patch_result.get("status") not in {"patched", "already_patched"}:
                        status = "degraded"
                        message = (
                            "Native runtime source-build repair stopped because the guarded cumm CUDA resolver patch failed; spconv install was skipped."
                        )
                        next_action = cuda_resolver_patch_result.get(
                            "next_action",
                            "Inspect the managed cumm/common.py CUDA resolver and retry native_mode=install explicitly before attempting spconv.",
                        )
                        break
                if step.package == "spconv" and plan.install_strategy == "linux-arm64-source-build":
                    validation_result = spconv_core_validator(managed_python=managed_python)
                    install_results.append(validation_result)
                    if not validation_result.get("ready", False):
                        status = "degraded"
                        message = (
                            "Native runtime source-build repair stopped because spconv.core_cc is not importable; final native readiness probe was skipped."
                        )
                        next_action = validation_result.get(
                            "next_action",
                            "Repair the spconv native build so spconv.core_cc imports cleanly before retrying native_mode=install.",
                        )
                        break
            if install_results and all(_native_install_result_succeeded(result) for result in install_results):
                final_probe = probe_native_runtime(managed_python=managed_python, plan=plan)
                status = final_probe.status
                message = final_probe.message
                next_action = final_probe.next_action

    return {
        "mode": resolved_mode,
        "ready": final_probe.ready,
        "status": status,
        "attempted_install": attempted_install,
        "install_strategy": plan.install_strategy,
        "host": plan.host.to_dict(),
        "plan": plan.to_dict(),
        "probe": final_probe.to_dict(),
        "install_results": install_results,
        "message": message,
        "next_action": next_action,
    }


def build_native_dependency_report(*, native_runtime: dict[str, Any]) -> dict[str, Any]:
    plan = native_runtime.get("plan", {})
    probe = native_runtime.get("probe") or {}
    install_results = native_runtime.get("install_results") or []
    return {
        "native_dependency_policy": {
            "platform": native_runtime.get("host", {}),
            "required_imports": plan.get("required_imports") or list(_native_required_imports()),
            "related_native_packages": plan.get("related_packages") or list(_native_related_packages()),
            "strategy": plan.get("status", "planned"),
            "install_strategy": plan.get("install_strategy", native_runtime.get("install_strategy", "unsupported")),
            "automatic_build_supported": bool(plan.get("automatic_install_supported", False)),
            "native_mode": native_runtime.get("mode", NATIVE_MODE_AUTO),
            "attempted_install": bool(native_runtime.get("attempted_install", False)),
            "install_steps": plan.get("install_steps") or [],
            "message": plan.get("message") or native_runtime.get("message"),
        },
        "native_dependency_blockers": {
            "platform": native_runtime.get("host", {}),
            "status": "clear" if native_runtime.get("ready", False) else native_runtime.get("status", "blocked"),
            "blocked_imports": probe.get("blocked_imports") or [],
            "missing_modules": probe.get("missing_modules") or [],
            "related_native_packages": plan.get("related_packages") or list(_native_related_packages()),
            "strategy": "guarded-explicit-install" if native_runtime.get("mode") == NATIVE_MODE_INSTALL else "plan-and-probe",
            "install_strategy": plan.get("install_strategy", native_runtime.get("install_strategy", "unsupported")),
            "attempted_install": bool(native_runtime.get("attempted_install", False)),
            "install_results": install_results,
            "message": native_runtime.get("message") or probe.get("message") or plan.get("message"),
            "next_action": native_runtime.get("next_action") or probe.get("next_action") or plan.get("next_action"),
        },
    }


def linux_arm64_torch_choice(*, py_tag: str, gpu_sm: int, cuda_version: int) -> dict[str, Any]:
    if gpu_sm >= 100 or cuda_version >= 128:
        wheel_map = ARM64_CU128_WHEELS
        label = "cu128"
        packages = ["torch==2.7.0", "torchvision==0.22.0"]
        index_url = "https://download.pytorch.org/whl/cu128"
    elif gpu_sm >= 70:
        wheel_map = ARM64_CU124_WHEELS
        label = "cu124"
        packages = ["torch==2.5.1", "torchvision==0.20.1"]
        index_url = "https://download.pytorch.org/whl/cu124"
    else:
        raise RuntimeError(
            "Linux ARM64 support currently requires NVIDIA CUDA with gpu_sm >= 70. "
            f"Received gpu_sm={gpu_sm} cuda_version={cuda_version}."
        )

    wheel_urls = wheel_map.get(py_tag)
    if wheel_urls is None:
        raise RuntimeError(
            "Unsupported Python tag for Linux ARM64 pinned PyTorch wheels: "
            f"{py_tag}. Supported tags: {', '.join(sorted(wheel_map))}"
        )

    return {
        "target": f"linux-arm64-{label}",
        "index_url": index_url,
        "extra_index_url": index_url,
        "packages": [wheel_urls["torch"], wheel_urls["torchvision"]],
        "pins": packages,
        "label": label,
    }


def torch_install_plan(*, system_name: str, machine: str, py_tag: str, gpu_sm: int, cuda_version: int) -> dict[str, Any]:
    if _is_linux_arm64(system_name, machine):
        return linux_arm64_torch_choice(py_tag=py_tag, gpu_sm=gpu_sm, cuda_version=cuda_version)

    if gpu_sm >= 100 or cuda_version >= 128:
        return {
            "target": "default-cu128",
            "index_url": "https://download.pytorch.org/whl/cu128",
            "packages": ["torch==2.7.0", "torchvision==0.22.0"],
            "label": "cu128",
        }
    if gpu_sm >= 70:
        return {
            "target": "default-cu124",
            "index_url": "https://download.pytorch.org/whl/cu124",
            "packages": ["torch==2.5.1", "torchvision==0.20.1"],
            "label": "cu124",
        }
    return {
        "target": "default-cu118",
        "index_url": "https://download.pytorch.org/whl/cu118",
        "packages": ["torch==2.5.1", "torchvision==0.20.1"],
        "label": "cu118",
    }


def verify_runtime(python_exe: str | Path) -> dict[str, Any]:
    script = (
        "import importlib, json, platform, sys; "
        f"mods={VERIFY_IMPORTS!r}; "
        "missing=[]; "
        "torch_imported=False; "
        "cuda_available=None; "
        "\nfor name in mods:\n"
        "    try:\n"
        "        module = importlib.import_module(name)\n"
        "        if name == 'torch':\n"
        "            torch_imported = True\n"
        "            cuda_available = bool(module.cuda.is_available())\n"
        "    except Exception:\n"
        "        missing.append(name)\n"
        "python_abi = getattr(sys, 'abiflags', '') or f'cp{sys.version_info.major}{sys.version_info.minor}'; "
        "print(json.dumps({'ready': not missing, 'missing': missing, 'python_exe': sys.executable, 'python_version': platform.python_version(), 'python_abi': python_abi, 'torch_imported': torch_imported, 'cuda_available': cuda_available}))"
    )
    result = subprocess.run([str(python_exe), "-c", script], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "ready": False,
            "missing": list(VERIFY_IMPORTS),
            "python_exe": str(python_exe),
            "python_version": None,
            "python_abi": None,
            "torch_imported": False,
            "cuda_available": None,
            "stderr": result.stderr.strip(),
        }
    payload = json.loads(result.stdout.strip())
    payload["python_exe"] = payload.get("python_exe") or str(python_exe)
    return payload


def _resolve_setup_payload(argv: list[str]) -> dict[str, Any]:
    if not argv:
        return {"ext_dir": str(ROOT)}

    first = argv[0]
    if first in {"setup", "apply", "install", "repair"}:
        argv = argv[1:]
        if not argv:
            return {"ext_dir": str(ROOT)}
        first = argv[0]

    payload = _parse_payload(first)
    if payload is not None:
        return payload

    resolved: dict[str, Any] = {"python_exe": first}
    resolved["ext_dir"] = argv[1] if len(argv) > 1 else str(ROOT)
    if len(argv) > 2:
        resolved["gpu_sm"] = argv[2]
    if len(argv) > 3:
        resolved["cuda_version"] = argv[3]
    if len(argv) > 4:
        resolved["args"] = argv[4:]
    return resolved


def _resolve_extension_dir(payload: dict[str, Any] | None = None) -> Path:
    source = payload or {}
    return Path(source.get("ext_dir") or ROOT).resolve()


def _resolve_gpu_sm(payload: dict[str, Any], gpu_runtime_details: dict[str, Any] | None = None) -> int:
    details = gpu_runtime_details or resolve_gpu_runtime_details(payload)
    return int(details.get("gpu_sm") or 0)


def _resolve_cuda_version(payload: dict[str, Any], gpu_runtime_details: dict[str, Any] | None = None) -> int:
    details = gpu_runtime_details or resolve_gpu_runtime_details(payload)
    return int(details.get("cuda_version") or 0)


def _weight_status(project_root: Path) -> dict[str, Any]:
    context = resolve_runtime_context(project_root=project_root)
    generator = Hunyuan3DPartGenerator(project_root=project_root, runtime_context=context)
    weight_path = generator._weight_path()
    ready = weight_path.is_file()
    return {
        "weight_owner_id": generator.weight_owner_id,
        "hf_repo": generator.hf_repo,
        "download_check": generator.download_check,
        "model_root": str(context.paths.model_root),
        "path": str(weight_path),
        "ready": ready,
        "status": "ready" if ready else "missing",
    }


def _managed_host_support(
    *,
    fallback_context: Any,
    runtime_verification: dict[str, Any],
    has_venv: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    host_payload = fallback_context.host_facts.to_dict()
    host_payload["python_version"] = runtime_verification.get("python_version") or host_payload["python_version"]
    host_payload["python_abi"] = runtime_verification.get("python_abi") or host_payload["python_abi"]

    support_payload = fallback_context.support.to_dict()
    required_dependencies = {
        module: module not in set(runtime_verification.get("missing") or [])
        for module in ("numpy", "torch", "trimesh")
    }
    support_payload["dependencies"] = required_dependencies

    if has_venv and runtime_verification.get("torch_imported") and runtime_verification.get("cuda_available") is not None:
        host_payload["cuda_visible"] = bool(runtime_verification["cuda_available"])
        host_payload["cuda_visible_source"] = "managed_python_torch"
        managed_support = evaluate_host_support(
            HostFacts(
                os_name=str(host_payload["os_name"]),
                arch=str(host_payload["arch"]),
                python_version=str(host_payload["python_version"]),
                python_abi=str(host_payload["python_abi"]),
                cuda_visible=bool(host_payload["cuda_visible"]),
            ),
            dependency_checker=lambda module: object() if required_dependencies.get(module, False) else None,
        ).to_dict()
        managed_support["dependencies"] = required_dependencies
        managed_support["source"] = "managed_python_torch"
        managed_support["observed_python"] = runtime_verification.get("python_exe")
        return host_payload, managed_support

    host_payload["cuda_visible_source"] = "current_process_env" if not has_venv else "managed_python_probe_pending"
    support_payload["source"] = "current_process_env" if not has_venv else "managed_python_probe_pending"
    if has_venv:
        support_payload.update(
            {
                "ready": False,
                "status": "pending",
                "warnings": list(support_payload.get("warnings") or []),
                "failure": {
                    "message": "Managed runtime verification did not produce a torch CUDA visibility result.",
                    "code": "managed_cuda_probe_pending",
                    "details": {
                        "python_exe": runtime_verification.get("python_exe") or host_payload.get("python_exe"),
                        "missing": list(runtime_verification.get("missing") or []),
                    },
                },
                "observed_python": runtime_verification.get("python_exe"),
            }
        )
    return host_payload, support_payload


def _collect_blocked_reasons(readiness: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    base_runtime = readiness["base_runtime"]
    upstream_runtime = readiness["upstream_runtime"]
    native_runtime = readiness["native_runtime"]
    runtime_adapter = readiness["runtime_adapter"]
    import_smoke = readiness["adapter_import_smoke"]
    weights = readiness["weights"]
    support = readiness["support"]

    if not base_runtime["ready"]:
        reasons.append("Managed extension venv/runtime packages are not ready yet.")
    if not upstream_runtime["ready"]:
        reasons.append("Upstream P3-SAM runtime source is not prepared yet.")
    if not native_runtime["ready"]:
        reasons.append(native_runtime.get("message") or "Native runtime dependencies are not ready yet.")
    if not weights["ready"]:
        reasons.append(f"Missing required model weights: {weights['path']}")
    if not runtime_adapter["ready"] and not import_smoke.get("blocked_by_native_runtime") and import_smoke.get("status") != "ready":
        reasons.append("Managed adapter import smoke is not ready yet.")
    if support.get("status") == "blocked" and support.get("failure", {}).get("code") != "cuda_not_visible":
        reasons.append(str(support["failure"].get("message") or "Host support requirements are not satisfied."))
    return reasons


def patch_prepared_upstream_source(extension_dir: Path) -> dict[str, Any]:
    runtime_root = runtime_source_root(extension_dir)
    cache_root = sonata_cache_root(extension_dir)

    p3_sam_patch = _patch_prepared_p3_sam_model_source(
        runtime_root / UPSTREAM_P3_SAM_MODEL_RELATIVE_PATH,
        cache_root=cache_root,
    )
    sonata_flash_attention_patch = _patch_prepared_sonata_model_source(
        runtime_root / UPSTREAM_SONATA_MODEL_RELATIVE_PATH,
    )
    components = {
        "p3_sam_sonata_cache_patch": p3_sam_patch,
        "sonata_flash_attention_patch": sonata_flash_attention_patch,
    }
    statuses = [component["status"] for component in components.values()]
    if any(status == "invalid" for status in statuses):
        status = "invalid"
    elif any(status == "missing" for status in statuses):
        status = "missing"
    elif any(status == "patched" for status in statuses):
        status = "patched"
    else:
        status = "already_patched"

    return {
        "ready": all(bool(component.get("ready", False)) for component in components.values()),
        "status": status,
        "cache_root": str(cache_root),
        "env_var": SONATA_CACHE_ENV_VAR,
        "paths": {
            "p3_sam_model": str(runtime_root / UPSTREAM_P3_SAM_MODEL_RELATIVE_PATH),
            "sonata_model": str(runtime_root / UPSTREAM_SONATA_MODEL_RELATIVE_PATH),
        },
        "components": components,
    }


def _patch_prepared_p3_sam_model_source(model_path: Path, *, cache_root: Path) -> dict[str, Any]:
    if not model_path.is_file():
        return {
            "ready": False,
            "status": "missing",
            "path": str(model_path),
            "cache_root": str(cache_root),
            "reason": "missing_model_py",
        }

    source = model_path.read_text(encoding="utf-8")
    patched_source = source
    status = "already_patched"
    if UPSTREAM_MODEL_PATCH_NEW not in source:
        if UPSTREAM_MODEL_PATCH_OLD not in source:
            raise RuntimeError(
                f"Unable to patch upstream P3-SAM model.py because the expected Sonata download root was not found: {model_path}"
            )
        patched_source = source.replace(UPSTREAM_MODEL_PATCH_OLD, UPSTREAM_MODEL_PATCH_NEW)
        status = "patched"

    if patched_source != source:
        model_path.write_text(patched_source, encoding="utf-8")

    verified = UPSTREAM_MODEL_PATCH_NEW in model_path.read_text(encoding="utf-8")
    return {
        "ready": verified,
        "status": status if verified else "invalid",
        "path": str(model_path),
        "cache_root": str(cache_root),
        "env_var": SONATA_CACHE_ENV_VAR,
    }


def _patch_prepared_sonata_model_source(model_path: Path) -> dict[str, Any]:
    if not model_path.is_file():
        return {
            "ready": False,
            "status": "missing",
            "path": str(model_path),
            "reason": "missing_sonata_model_py",
        }

    source = model_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    load_patch = _patch_sonata_flash_attention_load_site(lines, model_path=model_path)
    load_by_config_patch = _patch_sonata_flash_attention_load_by_config_site(lines, model_path=model_path)
    patches = (load_patch, load_by_config_patch)
    if any(patch["status"] == "invalid" for patch in patches):
        return {
            "ready": False,
            "status": "invalid",
            "path": str(model_path),
            "reason": "unexpected_sonata_flash_attention_patch_shape",
            "sites": {
                "load": load_patch,
                "load_by_config": load_by_config_patch,
            },
        }

    if any(patch["status"] == "patched" for patch in patches):
        model_path.write_text("\n".join(lines) + ("\n" if source.endswith("\n") else ""), encoding="utf-8")

    verified_lines = model_path.read_text(encoding="utf-8").splitlines()
    verified = _verify_sonata_flash_attention_sites(verified_lines)
    statuses = [patch["status"] for patch in patches]
    if any(status == "patched" for status in statuses):
        status = "patched"
    elif all(status == "already_patched" for status in statuses):
        status = "already_patched"
    else:
        status = "invalid"

    return {
        "ready": verified,
        "status": status if verified else "invalid",
        "path": str(model_path),
        "line": SONATA_FLASH_ATTENTION_LOAD_PATCH_NEW,
        "sites": {
            "load": load_patch,
            "load_by_config": load_by_config_patch,
        },
    }


def _patch_sonata_flash_attention_load_site(lines: list[str], *, model_path: Path) -> dict[str, Any]:
    return _patch_sonata_flash_attention_site(
        lines,
        model_path=model_path,
        model_line=SONATA_FLASH_ATTENTION_LOAD_MODEL_LINE,
        patch_line=SONATA_FLASH_ATTENTION_LOAD_PATCH_NEW,
        old_patch_line=SONATA_FLASH_ATTENTION_LOAD_PATCH_OLD,
        site="load",
    )


def _patch_sonata_flash_attention_load_by_config_site(lines: list[str], *, model_path: Path) -> dict[str, Any]:
    return _patch_sonata_flash_attention_site(
        lines,
        model_path=model_path,
        model_line=SONATA_FLASH_ATTENTION_LOAD_BY_CONFIG_MODEL_LINE,
        patch_line=SONATA_FLASH_ATTENTION_LOAD_BY_CONFIG_PATCH_NEW,
        expected_previous_line=SONATA_FLASH_ATTENTION_LOAD_BY_CONFIG_PREVIOUS_LINE,
        site="load_by_config",
    )


def _patch_sonata_flash_attention_site(
    lines: list[str],
    *,
    model_path: Path,
    model_line: str,
    patch_line: str,
    site: str,
    old_patch_line: str | None = None,
    expected_previous_line: str | None = None,
) -> dict[str, Any]:
    model_lines = [index for index, line in enumerate(lines) if line.strip() == model_line]
    if len(model_lines) != 1:
        return {
            "ready": False,
            "status": "invalid",
            "path": str(model_path),
            "reason": "unexpected_sonata_flash_attention_patch_shape",
            "line": patch_line,
            "site": site,
        }

    model_index = model_lines[0]
    if model_index == 0:
        return {
            "ready": False,
            "status": "invalid",
            "path": str(model_path),
            "reason": "unexpected_sonata_flash_attention_patch_shape",
            "line": patch_line,
            "site": site,
        }

    patch_line_index = model_index - 1
    while patch_line_index >= 0 and not lines[patch_line_index].strip():
        patch_line_index -= 1
    if patch_line_index < 0:
        return {
            "ready": False,
            "status": "invalid",
            "path": str(model_path),
            "reason": "unexpected_sonata_flash_attention_patch_shape",
            "line": patch_line,
            "site": site,
        }

    previous_line = lines[patch_line_index].strip()
    if previous_line == patch_line:
        return {
            "ready": True,
            "status": "already_patched",
            "path": str(model_path),
            "line": patch_line,
            "site": site,
        }

    if old_patch_line is not None:
        if previous_line != old_patch_line:
            return {
                "ready": False,
                "status": "invalid",
                "path": str(model_path),
                "reason": "unexpected_sonata_flash_attention_patch_shape",
                "line": patch_line,
                "site": site,
            }
        indent_match = re.match(r"^(\s*)", lines[model_index])
        indent = indent_match.group(1) if indent_match else ""
        lines[patch_line_index] = f"{indent}{patch_line}"
        return {
            "ready": True,
            "status": "patched",
            "path": str(model_path),
            "line": patch_line,
            "site": site,
        }

    if expected_previous_line is None or previous_line != expected_previous_line:
        return {
            "ready": False,
            "status": "invalid",
            "path": str(model_path),
            "reason": "unexpected_sonata_flash_attention_patch_shape",
            "line": patch_line,
            "site": site,
        }

    indent_match = re.match(r"^(\s*)", lines[model_index])
    indent = indent_match.group(1) if indent_match else ""
    lines.insert(model_index, f"{indent}{patch_line}")
    return {
        "ready": True,
        "status": "patched",
        "path": str(model_path),
        "line": patch_line,
        "site": site,
    }


def _verify_sonata_flash_attention_sites(lines: list[str]) -> bool:
    return _verify_sonata_flash_attention_site(
        lines,
        model_line=SONATA_FLASH_ATTENTION_LOAD_MODEL_LINE,
        patch_line=SONATA_FLASH_ATTENTION_LOAD_PATCH_NEW,
    ) and _verify_sonata_flash_attention_site(
        lines,
        model_line=SONATA_FLASH_ATTENTION_LOAD_BY_CONFIG_MODEL_LINE,
        patch_line=SONATA_FLASH_ATTENTION_LOAD_BY_CONFIG_PATCH_NEW,
    )


def _verify_sonata_flash_attention_site(lines: list[str], *, model_line: str, patch_line: str) -> bool:
    model_lines = [index for index, line in enumerate(lines) if line.strip() == model_line]
    if len(model_lines) != 1:
        return False
    patch_line_index = model_lines[0] - 1
    while patch_line_index >= 0 and not lines[patch_line_index].strip():
        patch_line_index -= 1
    return patch_line_index >= 0 and lines[patch_line_index].strip() == patch_line


def prepare_upstream_runtime_source(
    extension_dir: Path,
    *,
    repo_url: str = UPSTREAM_REPO_URL,
    repo_ref: str = UPSTREAM_REPO_REF,
) -> dict[str, Any]:
    runtime_root = runtime_source_root(extension_dir)
    runtime_root.parent.mkdir(parents=True, exist_ok=True)
    required_paths = [runtime_root / relative_path for relative_path in UPSTREAM_REQUIRED_PATHS]
    if all(path.exists() for path in required_paths):
        patch_prepared_upstream_source(extension_dir)
        return describe_upstream_runtime_source(extension_dir)

    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--sparse",
            "--depth",
            "1",
            "--branch",
            repo_ref,
            repo_url,
            str(runtime_root),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(runtime_root), "sparse-checkout", "set", "P3-SAM", "XPart/partgen"],
        check=True,
    )
    patch_prepared_upstream_source(extension_dir)
    return describe_upstream_runtime_source(extension_dir)


def describe_upstream_runtime_source(extension_dir: Path) -> dict[str, Any]:
    runtime_root = runtime_source_root(extension_dir)
    required_paths = [runtime_root / relative_path for relative_path in UPSTREAM_REQUIRED_PATHS]
    patch = patch_prepared_upstream_source(extension_dir)
    return {
        "ready": all(path.exists() for path in required_paths) and bool(patch.get("ready", False)),
        "status": "prepared" if all(path.exists() for path in required_paths) and bool(patch.get("ready", False)) else "missing",
        "source": str(runtime_root),
        "repo_url": UPSTREAM_REPO_URL,
        "repo_ref": UPSTREAM_REPO_REF,
        "required_paths": [str(path.relative_to(runtime_root)) for path in required_paths],
        "patch": patch,
    }


def build_readiness_report(
    *,
    project_root: Path = ROOT,
    gpu_sm: int = 0,
    cuda_version: int = 0,
    native_mode: str = NATIVE_MODE_AUTO,
    native_runtime: dict[str, Any] | None = None,
    gpu_runtime_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = resolve_runtime_context(project_root=project_root)
    system_name, machine = _platform_facts()
    venv_dir = project_root / "venv"
    managed_python = venv_python(venv_dir)
    has_venv = managed_python.exists()
    runtime_python = managed_python if has_venv else Path(sys.executable)
    py_tag = python_tag(runtime_python)
    gpu_runtime_details = gpu_runtime_details or {
        "gpu_sm": gpu_sm,
        "cuda_version": cuda_version,
        "gpu_sm_detail": {"field": "gpu_sm", "value": gpu_sm, "source": "provided"},
        "cuda_version_detail": {"field": "cuda_version", "value": cuda_version, "source": "provided"},
        "nvidia_smi": {"status": "not-requested", "errors": [], "commands": {}},
    }
    runtime_verification = verify_runtime(runtime_python)
    host, support = _managed_host_support(
        fallback_context=context,
        runtime_verification=runtime_verification,
        has_venv=has_venv,
    )
    generator = Hunyuan3DPartGenerator(project_root=project_root, runtime_context=context)
    weights = _weight_status(project_root)
    upstream_runtime = describe_upstream_runtime_source(project_root)
    runtime_adapter = build_adapter_readiness(
        project_root=project_root,
        managed_python=managed_python,
        model_root=context.paths.model_root,
    )
    native_runtime = native_runtime or run_native_runtime_phase(
        venv_dir=venv_dir,
        managed_python=managed_python,
        system_name=system_name,
        machine=machine,
        py_tag=py_tag,
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
        native_mode=native_mode,
    )
    native_dependency_report = build_native_dependency_report(native_runtime=native_runtime)
    import_smoke = dict(runtime_adapter.components.get("import_smoke", {}))
    native_probe = native_runtime.get("probe") or {}
    blocked_by_native_runtime = bool(
        set(import_smoke.get("missing_modules") or []) & set(native_probe.get("blocked_imports") or [])
    ) or bool(set(import_smoke.get("native_blockers") or []) & {"spconv", "torch_scatter"})
    import_smoke["blocked_by_native_runtime"] = blocked_by_native_runtime
    if blocked_by_native_runtime:
        import_smoke["native_blockers"] = native_probe.get("blocked_imports") or import_smoke.get("native_blockers") or []

    base_runtime = {
        "ready": bool(has_venv and runtime_verification.get("ready", False)),
        "status": "ready" if has_venv and runtime_verification.get("ready", False) else ("pending" if not has_venv else "blocked"),
        "venv": {
            "path": str(venv_dir),
            "python": str(managed_python),
            "exists": has_venv,
            "runtime_python": str(runtime_python),
            "python_tag": py_tag,
        },
        "runtime_verification": runtime_verification,
    }
    phases = {
        "base_runtime": {"ready": base_runtime["ready"], "status": base_runtime["status"]},
        "upstream_runtime": {"ready": upstream_runtime["ready"], "status": upstream_runtime["status"]},
        "native_runtime": {"ready": native_runtime["ready"], "status": native_runtime["status"], "mode": native_runtime["mode"]},
        "weights": {"ready": weights["ready"], "status": weights["status"]},
        "runtime_adapter": {"ready": runtime_adapter.ready, "status": runtime_adapter.status},
        "adapter_import_smoke": {
            "ready": bool(import_smoke.get("ready", False)),
            "status": str(import_smoke.get("status", "missing")),
            "blocked_by_native_runtime": blocked_by_native_runtime,
        },
    }
    runtime_ready = bool(base_runtime["ready"] and upstream_runtime["ready"] and runtime_adapter.ready)
    native_runtime_ready = bool(native_runtime["ready"])
    weights_ready = bool(weights["ready"])
    adapter_ready = bool(runtime_adapter.ready)
    execution_ready = bool(runtime_ready and native_runtime_ready)

    if not has_venv:
        next_action = "Run managed setup/repair to create the extension venv."
    elif upstream_runtime["status"] != "prepared":
        next_action = "Run managed setup/repair again to sparse-checkout and patch the upstream runtime source."
    elif native_runtime["status"] in {"blocked", "unsupported"}:
        next_action = native_runtime["next_action"]
    elif blocked_by_native_runtime:
        next_action = native_runtime["next_action"]
    else:
        next_action = "Run managed setup/repair again if adapter import smoke is still blocked after native dependencies are present."

    return {
        "setup_contract": "python-root-setup-py",
        "surface_owner": "electron",
        "plan_only": False,
        "live_identity": "unconfirmed",
        "shell_ready": True,
        "native_runtime_ready": native_runtime_ready,
        "weights_ready": weights_ready,
        "adapter_ready": adapter_ready,
        "runtime_ready": runtime_ready,
        "execution_ready": execution_ready,
        "inference_ready": False,
        "status": "ready" if execution_ready else "pending",
        "extension_dir": str(project_root),
        "host": host,
        "gpu_detection": gpu_runtime_details,
        "defaults": context.default_params,
        "paths": context.paths.to_dict(),
        "support": support,
        "venv": base_runtime["venv"],
        "runtime_verification": runtime_verification,
        "base_runtime": base_runtime,
        "runtime_adapter": runtime_adapter.to_dict(),
        "adapter_import_smoke": import_smoke,
        "native_runtime": native_runtime,
        **native_dependency_report,
        "phases": phases,
        "upstream_runtime": upstream_runtime,
        "weights": weights,
        "generator": generator.readiness_status(),
        "blocked_reasons": _collect_blocked_reasons(
            {
                "base_runtime": base_runtime,
                "upstream_runtime": upstream_runtime,
                "native_runtime": native_runtime,
                "runtime_adapter": runtime_adapter.to_dict(),
                "adapter_import_smoke": import_smoke,
                "weights": weights,
                "support": support,
            }
        ),
        "next_action": next_action,
    }


def _write_summary(summary: dict[str, Any], extension_dir: Path) -> Path:
    summary_path = extension_dir / SETUP_SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def _adapter_blocked_only_by_missing_weights(runtime_adapter: dict[str, Any]) -> bool:
    components = runtime_adapter.get("components")
    if not isinstance(components, dict):
        return False

    required_ready_components = ("managed_python", "runtime_source", "entrypoint", "import_smoke")
    required_components = (*required_ready_components, "weights")
    if any(not isinstance(components.get(name), dict) for name in required_components):
        return False

    if any(not bool(components[name].get("ready", False)) for name in required_ready_components):
        return False

    weights_component = components["weights"]
    return not bool(weights_component.get("ready", False)) and str(weights_component.get("status", "")) == "missing"


def _setup_exit_code(readiness: dict[str, Any]) -> int:
    if bool(readiness.get("execution_ready")):
        return 0

    base_runtime = readiness.get("base_runtime") or {}
    upstream_runtime = readiness.get("upstream_runtime") or {}
    native_runtime = readiness.get("native_runtime") or {}
    runtime_adapter = readiness.get("runtime_adapter") or {}
    weights = readiness.get("weights") or {}

    if not bool(base_runtime.get("ready")) or not bool(upstream_runtime.get("ready")):
        return 1

    if native_runtime.get("mode") == NATIVE_MODE_INSTALL and not bool(native_runtime.get("ready", False)):
        return 1

    setup_ready_except_weights = (
        bool(base_runtime.get("ready"))
        and bool(upstream_runtime.get("ready"))
        and bool(native_runtime.get("ready"))
        and _adapter_blocked_only_by_missing_weights(runtime_adapter)
        and not bool(weights.get("ready"))
    )
    if setup_ready_except_weights:
        return 0

    return 0


def run_setup(argv: list[str], *, as_json: bool) -> int:
    payload = _resolve_setup_payload(argv)
    extension_dir = _resolve_extension_dir(payload)
    extension_dir.mkdir(parents=True, exist_ok=True)
    source_python = str(payload.get("python_exe") or sys.executable)
    venv_dir = extension_dir / "venv"
    native_mode = _resolve_native_mode(payload)

    subprocess.run([source_python, "-m", "venv", str(venv_dir)], check=True)
    pip_install(venv_dir, "--upgrade", "pip", "setuptools", "wheel")

    py_tag = python_tag(venv_python(venv_dir))
    system_name, machine = _platform_facts()
    gpu_runtime_details = resolve_gpu_runtime_details(payload)
    gpu_sm = _resolve_gpu_sm(payload, gpu_runtime_details)
    cuda_version = _resolve_cuda_version(payload, gpu_runtime_details)
    torch_plan = torch_install_plan(
        system_name=system_name,
        machine=machine,
        py_tag=py_tag,
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
    )
    pip_install(
        venv_dir,
        *torch_plan["packages"],
        index_url=torch_plan.get("index_url"),
        extra_index_url=torch_plan.get("extra_index_url"),
    )
    pip_install(venv_dir, *RUNTIME_PACKAGES)
    upstream_runtime = prepare_upstream_runtime_source(extension_dir)
    native_runtime = run_native_runtime_phase(
        venv_dir=venv_dir,
        managed_python=venv_python(venv_dir),
        system_name=system_name,
        machine=machine,
        py_tag=py_tag,
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
        native_mode=native_mode,
    )

    readiness = build_readiness_report(
        project_root=extension_dir,
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
        native_mode=native_mode,
        native_runtime=native_runtime,
        gpu_runtime_details=gpu_runtime_details,
    )
    summary = {
        "status": readiness["status"],
        "prepared_shell": True,
        "native_runtime_ready": readiness["native_runtime_ready"],
        "weights_ready": readiness["weights_ready"],
        "adapter_ready": readiness["adapter_ready"],
        "runtime_ready": readiness["runtime_ready"],
        "execution_ready": readiness["execution_ready"],
        "inference_ready": readiness["inference_ready"],
        "setup_contract": readiness["setup_contract"],
        "surface_owner": readiness["surface_owner"],
        "generator_class": "Hunyuan3DPartGenerator",
        "extension_dir": str(extension_dir),
        "python_exe": source_python,
        "venv_python": readiness["venv"]["python"],
        "gpu_sm": gpu_sm,
        "cuda_version": cuda_version,
        "gpu_detection": gpu_runtime_details,
        "native_mode": native_mode,
        "install_strategy": native_runtime.get("install_strategy", "unsupported"),
        "torch_target": torch_plan["target"],
        "torch_packages": torch_plan["packages"],
        "runtime_packages": list(RUNTIME_PACKAGES),
        "base_runtime": readiness["base_runtime"],
        "upstream_runtime": upstream_runtime,
        "native_runtime": native_runtime,
        "native_build_artifacts": summarize_native_build_artifacts(
            extension_dir=extension_dir,
            native_runtime=native_runtime,
        ),
        "adapter_import_smoke": readiness["adapter_import_smoke"],
        "phases": readiness["phases"],
        "native_dependency_policy": readiness["native_dependency_policy"],
        "native_dependency_blockers": readiness["native_dependency_blockers"],
        "readiness": readiness,
        "blocked_reasons": list(readiness["blocked_reasons"]),
        "next_action": readiness["next_action"],
    }
    summary_path = _write_summary(summary, extension_dir)
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"ready: wrote {summary_path}")
    return _setup_exit_code(readiness)


def main(argv: list[str] | None = None) -> int:
    raw_args = list(argv or sys.argv[1:])
    as_json = False
    if "--json" in raw_args:
        raw_args.remove("--json")
        as_json = True

    if not raw_args or raw_args[0] == "readiness":
        payload = _resolve_setup_payload(raw_args[1:]) if raw_args[:1] == ["readiness"] else {"ext_dir": str(ROOT)}
        gpu_runtime_details = resolve_gpu_runtime_details(payload)
        report = build_readiness_report(
            project_root=_resolve_extension_dir(payload),
            gpu_sm=_resolve_gpu_sm(payload, gpu_runtime_details),
            cuda_version=_resolve_cuda_version(payload, gpu_runtime_details),
            native_mode=_resolve_native_mode(payload),
            gpu_runtime_details=gpu_runtime_details,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    return run_setup(raw_args, as_json=as_json)


if __name__ == "__main__":
    raise SystemExit(main())
