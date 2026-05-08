from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_setup_module():
    spec = importlib.util.spec_from_file_location("hunyuan3d_part_setup", ROOT / "setup.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SetupContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setup = load_setup_module()

    def _create_fake_cumm_layout(self, root: Path) -> tuple[Path, Path, Path, Path, Path]:
        venv_dir = root / "venv"
        site_packages = venv_dir / "lib" / "python3.12" / "site-packages"
        package_root = site_packages / "cumm"
        package_root.mkdir(parents=True, exist_ok=True)
        (package_root / "constants.py").write_text(
            "PACKAGE_ROOT = __import__('pathlib').Path(__file__).parent.resolve()\n",
            encoding="utf-8",
        )
        expected_include = site_packages / "include"
        fallback_include = package_root / "include"
        sentinel = expected_include / "tensorview" / "tensor.h"
        return venv_dir, site_packages, package_root, expected_include, sentinel

    def _write_fake_cumm_common(self, site_packages: Path, source: str | None = None) -> Path:
        common_path = site_packages / "cumm" / "common.py"
        common_path.parent.mkdir(parents=True, exist_ok=True)
        common_path.write_text(source or self.setup.CUMM_CUDA_RESOLVER_LINUX_FRAGMENT, encoding="utf-8")
        return common_path

    def _write_fake_cumm_subint_header(self, include_root: Path, source: str | None = None) -> Path:
        header_path = include_root / "tensorview" / "gemm" / "dtypes" / "subint.h"
        header_path.parent.mkdir(parents=True, exist_ok=True)
        header_path.write_text(
            "#pragma once\n\nnamespace tv {\n\n"
            + (source or self.setup.CUMM_SUBINT_PATCH_BODY_ORIGINAL)
            + self.setup.CUMM_SUBINT_BODY_SUFFIX_MARKER
            + "\n",
            encoding="utf-8",
        )
        return header_path

    def _write_fake_cumm_tf32_header(self, include_root: Path, source: str | None = None) -> Path:
        header_path = include_root / "tensorview" / "gemm" / "dtypes" / "tf32.h"
        header_path.parent.mkdir(parents=True, exist_ok=True)
        header_path.write_text(
            source or self._legacy_tf32_header_source(),
            encoding="utf-8",
        )
        return header_path

    def _write_fake_cumm_half_header(self, include_root: Path, source: str | None = None) -> Path:
        header_path = include_root / "tensorview" / "gemm" / "dtypes" / "half.h"
        header_path.parent.mkdir(parents=True, exist_ok=True)
        header_path.write_text(
            source or self._live_like_half_header_source(),
            encoding="utf-8",
        )
        return header_path

    def _write_fake_cumm_bfloat16_header(self, include_root: Path, source: str | None = None) -> Path:
        header_path = include_root / "tensorview" / "gemm" / "dtypes" / "bfloat16.h"
        header_path.parent.mkdir(parents=True, exist_ok=True)
        header_path.write_text(
            source or self._live_like_bfloat16_header_source(),
            encoding="utf-8",
        )
        return header_path

    def _live_like_half_header_source(self) -> str:
        return """#pragma once
#include <tensorview/core/all.h>
#include <tensorview/dtypes.h>
#ifndef CUTLASS_ENABLE_F16C
#define CUTLASS_ENABLE_F16C 0
#endif


#if defined(__CUDACC_RTC__)
#include \"fp_nvrtc.h\"
#include <tensorview/core/nvrtc_std.h>
#undef CUTLASS_ENABLE_F16C
#define CUTLASS_ENABLE_F16C 0
#else
#include <cmath>
#include <cstdint>
#include <limits>
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
#if (__CUDACC_VER_MAJOR__ >= 10)
#include <cuda_fp16.h>
#endif
#endif
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace tv {

struct alignas(2) half_t {

  /// Constructs from an unsigned short
  TV_HOST_DEVICE_INLINE
  static half_t bitcast(uint16_t x) {
    half_t h;
    h.storage = x;
    return h;
  }

  /// Storage type
  uint16_t storage;

/// FP32 -> FP16 conversion - rounds to nearest even
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)
  // Avoid inlining in device code if no hardware support
  __device__ __noinline__
#else
  TV_HOST_DEVICE_INLINE
#endif
      static half_t
      convert(float const &flt) {
    return bitcast(static_cast<uint16_t>(flt));
  }

  /// Converts a half-precision value stored as a uint16_t to a float
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)
  // Avoid inlining in device code if no hardware support
  __device__ __noinline__
#else
  TV_HOST_DEVICE_INLINE
#endif
      static float
      convert(half_t const &x) {
    return static_cast<float>(x.storage);
  }

  TV_HOST_DEVICE_INLINE
  uint16_t raw() const { return storage; }

  TV_HOST_DEVICE_INLINE
  int exponent_biased() const { return int((storage >> 10) & 0x1f); }

  TV_HOST_DEVICE_INLINE
  int mantissa() const { return int(storage & 0x3ff); }
};

TV_HOST_DEVICE_INLINE
bool signbit(tv::half_t const &h) { return ((h.raw() & 0x8000) != 0); }

TV_HOST_DEVICE_INLINE
bool isnan(tv::half_t const &h) {
  return (h.exponent_biased() == 0x1f) && h.mantissa();
}

TV_HOST_DEVICE_INLINE
bool isfinite(tv::half_t const &h) { return (h.exponent_biased() != 0x1f); }

TV_HOST_DEVICE_INLINE
bool isinf(tv::half_t const &h) {
  return (h.exponent_biased() == 0x1f) && !h.mantissa();
}

TV_HOST_DEVICE_INLINE
bool isnormal(tv::half_t const &h) {
  return h.exponent_biased() && h.exponent_biased() != 0x1f;
}

TV_HOST_DEVICE_INLINE
int fpclassify(tv::half_t const &h) {
  int exp = h.exponent_biased();
  int mantissa = h.mantissa();
  if (exp == 0x1f) {
    if (mantissa) {
      return FP_NAN;
    } else {
      return FP_INFINITE;
    }
  } else if (!exp) {
    if (mantissa) {
      return FP_SUBNORMAL;
    } else {
      return FP_ZERO;
    }
  }
  return FP_NORMAL;
}

TV_HOST_DEVICE_INLINE
tv::half_t sqrt(tv::half_t const &h) {
#if defined(__CUDACC_RTC__)
  return tv::half_t(sqrtf(float(h)));
#else
  return tv::half_t(std::sqrt(float(h)));
#endif
}

TV_HOST_DEVICE_INLINE
half_t copysign(half_t const &a, half_t const &b) {

  uint16_t a_mag = (reinterpret_cast<uint16_t const &>(a) & 0x7fff);
  uint16_t b_sign = (reinterpret_cast<uint16_t const &>(b) & 0x8000);
  uint16_t result = (a_mag | b_sign);

  return reinterpret_cast<tv::half_t const &>(result);
}

} // namespace tv

namespace std {
template <> struct numeric_limits<tv::half_t> {};
}
"""

    def _patched_half_header_source(self) -> str:
        patched, status, reason = self.setup._canonicalize_cumm_half_header(self._live_like_half_header_source())
        self.assertEqual(status, "patched")
        self.assertIsNone(reason)
        return patched

    def _malformed_half_header_source(self) -> str:
        return self._live_like_half_header_source().replace(
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
  static __device__ __noinline__ half_t convert(float const &flt) {
#else
  static TV_HOST_DEVICE_INLINE half_t convert(float const &flt) {
#endif
""",
            1,
        ).replace(
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
  static __device__ __noinline__ float convert(half_t const &x) {
#else
  static TV_HOST_DEVICE_INLINE float convert(half_t const &x) {
#endif
""",
            1,
        )

    def _live_like_bfloat16_header_source(self) -> str:
        return """#pragma once
#if !defined(__CUDACC_RTC__)
#include <cmath>
#include <cstdint>
#include <limits>
#else 
#include \"fp_nvrtc.h\"
#include <tensorview/core/nvrtc_std.h>
#endif
#include \"half.h\"
#include <tensorview/core/all.h>

#ifdef __CUDACC__
#if (__CUDACC_VER_MAJOR__ >= 11)
#include <cuda_bf16.h>
#endif
#endif

namespace tv {

struct alignas(2) bfloat16_t {
  uint16_t storage;

  TV_HOST_DEVICE_INLINE
  bool signbit() const { return ((storage & 0x8000) != 0); }

  TV_HOST_DEVICE_INLINE
  int exponent_biased() const { return int((storage >> 7) & 0x0ff); }

  TV_HOST_DEVICE_INLINE
  int mantissa() const { return int(storage & 0x7f); }
};

TV_HOST_DEVICE_INLINE
bool signbit(tv::bfloat16_t const &h) { return h.signbit(); }

TV_HOST_DEVICE_INLINE
bool isnan(tv::bfloat16_t const &h) {
  return (h.exponent_biased() == 0x0ff) && h.mantissa();
}

TV_HOST_DEVICE_INLINE
bool isfinite(tv::bfloat16_t const &h) {
  return (h.exponent_biased() != 0x0ff);
}

TV_HOST_DEVICE_INLINE
bool isinf(tv::bfloat16_t const &h) {
  return (h.exponent_biased() == 0x0ff) && !h.mantissa();
}

TV_HOST_DEVICE_INLINE
bool isnormal(tv::bfloat16_t const &h) {
  return h.exponent_biased() && h.exponent_biased() != 0x0ff;
}

TV_HOST_DEVICE_INLINE
int fpclassify(tv::bfloat16_t const &h) {
  int exp = h.exponent_biased();
  int mantissa = h.mantissa();
  if (exp == 0x0ff) {
    if (mantissa) {
      return FP_NAN;
    } else {
      return FP_INFINITE;
    }
  } else if (!exp) {
    if (mantissa) {
      return FP_SUBNORMAL;
    } else {
      return FP_ZERO;
    }
  }
  return FP_NORMAL;
}

TV_HOST_DEVICE_INLINE
tv::bfloat16_t sqrt(tv::bfloat16_t const &h) {
#if defined(__CUDACC_RTC__)
  return tv::bfloat16_t(sqrtf(float(h)));
#else
  return tv::bfloat16_t(std::sqrt(float(h)));
#endif
}

TV_HOST_DEVICE_INLINE
bfloat16_t copysign(bfloat16_t const &a, bfloat16_t const &b) {

  uint16_t a_mag = (reinterpret_cast<uint16_t const &>(a) & 0x7fff);
  uint16_t b_sign = (reinterpret_cast<uint16_t const &>(b) & 0x8000);
  uint16_t result = (a_mag | b_sign);

  return reinterpret_cast<bfloat16_t const &>(result);
}

} // namespace tv

namespace std {
template <> struct numeric_limits<tv::bfloat16_t> {};
}
"""

    def _patched_bfloat16_header_source(self) -> str:
        patched, status, reason = self.setup._canonicalize_cumm_bfloat16_header(self._live_like_bfloat16_header_source())
        self.assertEqual(status, "patched")
        self.assertIsNone(reason)
        return patched

    def _legacy_tf32_header_source(self) -> str:
        return self.setup._read_cumm_header_snapshot(
            self.setup.CUMM_DTYPE_HEADER_LEGACY_VERSION,
            self.setup.CUMM_TF32_HEADER_RELATIVE_PATH,
        )

    def _canonical_tf32_header_source(self) -> str:
        return self.setup._read_cumm_header_snapshot(
            self.setup.CUMM_DTYPE_HEADER_CANONICAL_VERSION,
            self.setup.CUMM_TF32_HEADER_RELATIVE_PATH,
        )

    def _local_target_tf32_header_source(self) -> str:
        return self.setup._local_target_cumm_dtype_header_source(
            name="tf32",
            relative_path=self.setup.CUMM_TF32_HEADER_RELATIVE_PATH,
            legacy_fragments=self.setup.CUMM_TF32_HEADER_CANONICAL_FRAGMENTS,
        )

    def _previous_local_target_tf32_header_source(self) -> str:
        return self.setup._previous_local_target_tf32_header_source(
            self._canonical_tf32_header_source(),
        )

    def _latest_live_broken_tf32_header_source(self) -> str:
        return self.setup._latest_live_broken_tf32_header_source(
            self._canonical_tf32_header_source(),
        )

    def _write_fake_cumm_tf32_malformed_patched_header(self, include_root: Path) -> Path:
        return self._write_fake_cumm_tf32_header(
            include_root,
            source=self.setup._legacy_tf32_malformed_variants(self._legacy_tf32_header_source())[0],
        )

    def _write_fake_cumm_tf32_partially_patched_header(self, include_root: Path) -> Path:
        return self._write_fake_cumm_tf32_header(
            include_root,
            source=self.setup._legacy_tf32_malformed_variants(self._legacy_tf32_header_source())[-1],
        )

    def _write_fake_cumm_float8_header(self, include_root: Path, source: str | None = None) -> Path:
        header_path = include_root / "tensorview" / "gemm" / "dtypes" / "float8.h"
        header_path.parent.mkdir(parents=True, exist_ok=True)
        header_path.write_text(
            source or self._legacy_float8_header_source(),
            encoding="utf-8",
        )
        return header_path

    def _legacy_float8_header_source(self) -> str:
        return self.setup._read_cumm_header_snapshot(
            self.setup.CUMM_DTYPE_HEADER_LEGACY_VERSION,
            self.setup.CUMM_FLOAT8_HEADER_RELATIVE_PATH,
        )

    def _canonical_float8_header_source(self) -> str:
        return self.setup._read_cumm_header_snapshot(
            self.setup.CUMM_DTYPE_HEADER_CANONICAL_VERSION,
            self.setup.CUMM_FLOAT8_HEADER_RELATIVE_PATH,
        )

    def _local_target_float8_header_source(self) -> str:
        return self.setup._local_target_cumm_dtype_header_source(
            name="float8",
            relative_path=self.setup.CUMM_FLOAT8_HEADER_RELATIVE_PATH,
            legacy_fragments=self.setup.CUMM_FLOAT8_HEADER_CANONICAL_FRAGMENTS,
        )

    def _legacy_local_target_float8_header_source(self) -> str:
        return self.setup._legacy_local_target_cumm_dtype_header_source(
            canonical_source=self._canonical_float8_header_source(),
            name="float8",
            legacy_fragments=self.setup.CUMM_FLOAT8_HEADER_CANONICAL_FRAGMENTS,
        )

    def _previous_local_target_float8_header_source(self) -> str:
        return self.setup._previous_local_target_float8_header_source(
            self._canonical_float8_header_source(),
        )

    def _latest_live_broken_float8_header_source(self) -> str:
        return self.setup._latest_live_broken_float8_header_source(
            self._canonical_float8_header_source(),
        )

    def _approximate_guarded_float8_header_source(self) -> str:
        source = self.setup._legacy_float8_approximate_variant(
            self._canonical_float8_header_source(),
        )
        self.assertIsNotNone(source)
        assert source is not None
        return source

    def _write_fake_cumm_float8_malformed_patched_header(self, include_root: Path) -> Path:
        return self._write_fake_cumm_float8_header(
            include_root,
            source=self.setup._legacy_float8_malformed_variants(self._legacy_float8_header_source())[-1],
        )

    def _write_fake_cumm_float8_partially_patched_header(self, include_root: Path) -> Path:
        return self._write_fake_cumm_float8_header(
            include_root,
            source=self.setup._legacy_float8_malformed_variants(self._legacy_float8_header_source())[0],
        )

    def _extract_fake_cumm_subint_body(self, source: str) -> str:
        split = self.setup._split_cumm_subint_header_sections(source)
        self.assertIsNotNone(split)
        assert split is not None
        return split[1]

    def _assert_balanced_braces(self, text: str) -> None:
        balance = 0
        for char in text:
            if char == "{":
                balance += 1
            elif char == "}":
                balance -= 1
            self.assertGreaterEqual(balance, 0)
        self.assertEqual(balance, 0)

    def _assert_no_free_floating_subint_operators(self, body: str) -> None:
        struct_end = body.index("\n};")
        trailing_after_struct = body[struct_end + 3 :]
        self.assertNotIn("operator==(self_type const &rhs)", trailing_after_struct)
        self.assertNotIn("operator!=(self_type const &rhs)", trailing_after_struct)
        self.assertNotIn("operator<=(self_type const &rhs)", trailing_after_struct)
        self.assertNotIn("operator<(self_type const &rhs)", trailing_after_struct)
        self.assertNotIn("operator>=(self_type const &rhs)", trailing_after_struct)
        self.assertNotIn("operator>(self_type const &rhs)", trailing_after_struct)

    def _assert_no_custom_subint_comparison_operators(self, body: str) -> None:
        self.assertNotIn("operator==(self_type const &rhs)", body)
        self.assertNotIn("operator!=(self_type const &rhs)", body)
        self.assertNotIn("operator<=(self_type const &rhs)", body)
        self.assertNotIn("operator<(self_type const &rhs)", body)
        self.assertNotIn("operator>=(self_type const &rhs)", body)
        self.assertNotIn("operator>(self_type const &rhs)", body)

    def _float8_static_constants_region(self, source: str) -> str:
        start = source.index("    static constexpr int FP8_NUM_BITS = 8;\n")
        end = source.index("\n    // 256 in float\n")
        return source[start:end]

    def _assert_native_path_order(self, path_value: str, *, venv_dir: Path, cuda_bin: str, existing_path: str) -> None:
        self.assertEqual(path_value, f"{venv_dir / 'bin'}:{cuda_bin}:{existing_path}")

    def _assert_cuda_env_targets_selected_toolkit(self, env: dict[str, str], *, toolkit_root: str) -> None:
        include_dir = f"{toolkit_root}/include"
        nvcc_path = f"{toolkit_root}/bin/nvcc"
        self.assertEqual(env["CUDA_HOME"], toolkit_root)
        self.assertEqual(env["CUDA_PATH"], toolkit_root)
        self.assertEqual(env["CUDA_ROOT"], toolkit_root)
        self.assertEqual(env["CUDA_BIN_PATH"], toolkit_root)
        self.assertEqual(env["CUDA_TOOLKIT_ROOT_DIR"], toolkit_root)
        self.assertEqual(env["CUDACXX"], nvcc_path)
        self.assertEqual(env["CMAKE_CUDA_COMPILER"], nvcc_path)
        self.assertEqual(env["CUDA_NVCC_EXECUTABLE"], nvcc_path)
        self.assertTrue(env["CPATH"].startswith(include_dir))
        self.assertTrue(env["C_INCLUDE_PATH"].startswith(include_dir))
        self.assertTrue(env["CPLUS_INCLUDE_PATH"].startswith(include_dir))
        self.assertTrue(env["CPPFLAGS"].startswith(f"-isystem {include_dir}"))
        self.assertTrue(env["CFLAGS"].startswith(f"-isystem {include_dir}"))
        self.assertTrue(env["CXXFLAGS"].startswith(f"-isystem {include_dir}"))
        self.assertTrue(env["NVCCFLAGS"].startswith(f"-I{include_dir}"))
        self.assertTrue(env["CUDAFLAGS"].startswith(f"-I{include_dir}"))
        self.assertNotIn("/usr/local/cuda/include", env["CPATH"])
        self.assertNotIn("/usr/local/cuda/include", env["C_INCLUDE_PATH"])
        self.assertNotIn("/usr/local/cuda/include", env["CPLUS_INCLUDE_PATH"])

    def test_resolve_setup_payload_accepts_json_shape(self) -> None:
        payload = self.setup._resolve_setup_payload(
            [json.dumps({"python_exe": "/opt/python", "ext_dir": "/tmp/ext", "gpu_sm": 89})]
        )
        self.assertEqual(payload["python_exe"], "/opt/python")
        self.assertEqual(payload["ext_dir"], "/tmp/ext")
        self.assertEqual(payload["gpu_sm"], 89)

    def test_resolve_setup_payload_accepts_positional_shape(self) -> None:
        payload = self.setup._resolve_setup_payload(
            ["/usr/bin/python3", "/tmp/ext", "100", "128", "native_mode=install"]
        )
        self.assertEqual(payload["python_exe"], "/usr/bin/python3")
        self.assertEqual(payload["ext_dir"], "/tmp/ext")
        self.assertEqual(payload["gpu_sm"], "100")
        self.assertEqual(payload["cuda_version"], "128")
        self.assertEqual(payload["args"], ["native_mode=install"])

    def test_resolve_native_mode_accepts_json_and_positional_shapes(self) -> None:
        self.assertEqual(self.setup._resolve_native_mode({"native_mode": "install"}), "install")
        self.assertEqual(self.setup._resolve_native_mode({"args": ["--native-mode", "skip"]}), "skip")
        self.assertEqual(self.setup._resolve_native_mode({"args": ["native_mode=auto"]}), "auto")

    def test_resolve_native_mode_accepts_environment_overrides(self) -> None:
        with mock.patch.dict(
            self.setup.os.environ,
            {"HUNYUAN3D_PART_NATIVE_MODE": "install"},
            clear=False,
        ):
            self.assertEqual(self.setup._resolve_native_mode({}), "install")

        with mock.patch.dict(
            self.setup.os.environ,
            {"P3SAM_NATIVE_MODE": "skip"},
            clear=False,
        ):
            self.assertEqual(self.setup._resolve_native_mode({}), "skip")

    def test_parse_highest_compute_capability_prefers_highest_gpu(self) -> None:
        parsed = self.setup._parse_highest_compute_capability("8.9\n12.1\n7.5\n")

        self.assertTrue(parsed["detected"])
        self.assertEqual(parsed["compute_capabilities"], ["8.9", "12.1", "7.5"])
        self.assertEqual(parsed["selected_compute_capability"], "12.1")
        self.assertEqual(parsed["gpu_sm"], 121)

    def test_parse_cuda_version_output_reads_nvidia_smi_version(self) -> None:
        parsed = self.setup._parse_cuda_version_output(
            "NVIDIA-SMI 580.142 Driver Version: 580.142 CUDA Version        : 13.0\n"
        )

        self.assertTrue(parsed["detected"])
        self.assertEqual(parsed["cuda_version_text"], "13.0")
        self.assertEqual(parsed["cuda_version"], 130)

    def test_explicit_payload_values_override_detection(self) -> None:
        with mock.patch.object(
            self.setup,
            "detect_gpu_runtime",
            return_value={
                "detected": True,
                "gpu_sm": 121,
                "cuda_version": 130,
                "selected_compute_capability": "12.1",
                "compute_capabilities": ["12.1"],
                "cuda_version_text": "13.0",
                "commands": {},
                "errors": [],
                "status": "detected",
            },
        ) as detect_mock:
            details = self.setup.resolve_gpu_runtime_details({"gpu_sm": 89, "cuda_version": 124})

        detect_mock.assert_not_called()
        self.assertEqual(details["gpu_sm"], 89)
        self.assertEqual(details["cuda_version"], 124)
        self.assertEqual(details["gpu_sm_detail"]["source"], "payload")
        self.assertEqual(details["cuda_version_detail"]["source"], "payload")

    def test_zero_or_missing_payload_uses_detection(self) -> None:
        with mock.patch.object(
            self.setup,
            "detect_gpu_runtime",
            return_value={
                "detected": True,
                "gpu_sm": 121,
                "cuda_version": 130,
                "selected_compute_capability": "12.1",
                "compute_capabilities": ["12.1"],
                "cuda_version_text": "13.0",
                "commands": {},
                "errors": [],
                "status": "detected",
            },
        ):
            details = self.setup.resolve_gpu_runtime_details({"gpu_sm": 0, "cuda_version": ""})

        self.assertEqual(details["gpu_sm"], 121)
        self.assertEqual(details["cuda_version"], 130)
        self.assertEqual(details["gpu_sm_detail"]["source"], "nvidia-smi")
        self.assertEqual(details["cuda_version_detail"]["source"], "nvidia-smi")

    def test_detection_failure_does_not_crash(self) -> None:
        with mock.patch.object(
            self.setup,
            "detect_gpu_runtime",
            return_value={
                "detected": False,
                "gpu_sm": 0,
                "cuda_version": 0,
                "selected_compute_capability": None,
                "compute_capabilities": [],
                "cuda_version_text": None,
                "commands": {},
                "errors": ["nvidia-smi not found"],
                "status": "unavailable",
            },
        ):
            details = self.setup.resolve_gpu_runtime_details({})
            gpu_sm = self.setup._resolve_gpu_sm({}, details)
            cuda_version = self.setup._resolve_cuda_version({}, details)

        self.assertEqual(gpu_sm, 0)
        self.assertEqual(cuda_version, 0)
        self.assertEqual(details["nvidia_smi"]["status"], "unavailable")
        self.assertEqual(details["gpu_sm_detail"]["detection_errors"], ["nvidia-smi not found"])

    def test_linux_arm64_torch_choice_prefers_cu128(self) -> None:
        choice = self.setup.linux_arm64_torch_choice(py_tag="cp311", gpu_sm=100, cuda_version=124)
        self.assertEqual(choice["target"], "linux-arm64-cu128")
        self.assertEqual(choice["label"], "cu128")
        self.assertEqual(len(choice["packages"]), 2)

    def test_resolve_cuda_toolkit_for_cu128_prefers_cuda_12_8(self) -> None:
        def fake_exists(path_self: Path) -> bool:
            return str(path_self) in {
                "/usr/local/cuda-12.8/bin/nvcc",
                "/usr/local/cuda-12.8/lib64",
                "/usr/local/cuda-12.8/targets/sbsa-linux/lib",
            }

        with (
            mock.patch.object(Path, "exists", fake_exists),
            mock.patch.object(self.setup, "_read_nvcc_release", return_value="12.8") as release_mock,
        ):
            toolkit = self.setup.resolve_cuda_toolkit_for_label("cu128")

        self.assertEqual(toolkit["status"], "matched")
        self.assertEqual(toolkit["toolkit_root"], "/usr/local/cuda-12.8")
        self.assertEqual(toolkit["include_dir"], "/usr/local/cuda-12.8/include")
        self.assertEqual(toolkit["nvcc_path"], "/usr/local/cuda-12.8/bin/nvcc")
        self.assertEqual(toolkit["nvcc_version"], "12.8")
        self.assertEqual(toolkit["lib_dirs"], [
            "/usr/local/cuda-12.8/lib64",
            "/usr/local/cuda-12.8/targets/sbsa-linux/lib",
        ])
        release_mock.assert_called_once()

    def test_resolve_cuda_toolkit_for_cu124_uses_default_only_when_nvcc_matches(self) -> None:
        def fake_exists(path_self: Path) -> bool:
            return str(path_self) in {
                "/usr/local/cuda/bin/nvcc",
                "/usr/local/cuda/lib64",
            }

        with (
            mock.patch.object(Path, "exists", fake_exists),
            mock.patch.object(self.setup, "_read_nvcc_release", return_value="12.4"),
        ):
            toolkit = self.setup.resolve_cuda_toolkit_for_label("cu124")

        self.assertEqual(toolkit["status"], "matched")
        self.assertEqual(toolkit["toolkit_root"], "/usr/local/cuda")
        self.assertEqual(toolkit["include_dir"], "/usr/local/cuda/include")
        self.assertEqual(toolkit["nvcc_path"], "/usr/local/cuda/bin/nvcc")
        self.assertEqual(toolkit["nvcc_version"], "12.4")

    def test_build_native_runtime_plan_marks_linux_arm64_lane_supported(self) -> None:
        venv_dir = Path("/tmp/native-plan-cu128")
        with (
            mock.patch.object(
                self.setup,
                "resolve_cuda_toolkit_for_label",
                return_value={
                    "desired_cuda_label": "cu128",
                    "toolkit_root": "/usr/local/cuda-12.8",
                    "include_dir": "/usr/local/cuda-12.8/include",
                    "nvcc_path": "/usr/local/cuda-12.8/bin/nvcc",
                    "nvcc_version": "12.8",
                    "status": "matched",
                    "reason": "ok",
                    "lib_dirs": ["/usr/local/cuda-12.8/lib64"],
                },
            ),
            mock.patch.dict(self.setup.os.environ, {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/usr/lib"}, clear=False),
        ):
            plan = self.setup.build_native_runtime_plan(
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                native_mode="auto",
                venv_dir=venv_dir,
            )

        self.assertEqual(plan.host.lane, "linux-arm64-cu128")
        self.assertEqual(plan.host.state, "supported")
        self.assertEqual(plan.status, "planned")
        self.assertTrue(plan.automatic_install_supported)
        self.assertEqual(plan.install_strategy, "linux-arm64-source-build")
        self.assertEqual(plan.desired_cuda_label, "cu128")
        self.assertEqual(plan.toolkit_root, "/usr/local/cuda-12.8")
        self.assertEqual(plan.toolkit_include_dir, "/usr/local/cuda-12.8/include")
        self.assertEqual(plan.toolkit_nvcc, "/usr/local/cuda-12.8/bin/nvcc")
        self.assertEqual(plan.toolkit_lib_dirs, ("/usr/local/cuda-12.8/lib64",))
        self.assertEqual(plan.nvcc_version, "12.8")
        self._assert_cuda_env_targets_selected_toolkit(plan.install_steps[0].env, toolkit_root="/usr/local/cuda-12.8")
        self._assert_native_path_order(
            plan.install_steps[0].env["PATH"],
            venv_dir=venv_dir,
            cuda_bin="/usr/local/cuda-12.8/bin",
            existing_path="/usr/bin",
        )
        self.assertEqual([step.package for step in plan.install_steps], [
            "native-build-prerequisites",
            "torch_scatter",
            "torch_cluster",
            "cumm",
            "spconv",
        ])
        self.assertEqual(plan.install_steps[1].env["TORCH_CUDA_ARCH_LIST"], "10.0")
        self.assertNotIn("CUMM_CUDA_ARCH_LIST", plan.install_steps[2].env)
        self.assertEqual(plan.install_steps[3].env["CUMM_DISABLE_JIT"], "1")
        self.assertIn("only validate explicit cumm arches through 9.0", plan.install_steps[3].reason)
        self.assertNotIn("TORCH_CUDA_ARCH_LIST", plan.install_steps[3].env)
        self.assertNotIn("CUMM_CUDA_ARCH_LIST", plan.install_steps[4].env)
        self.assertEqual(plan.install_steps[4].env["SPCONV_DISABLE_JIT"], "1")
        self.assertNotIn("CUMM_DISABLE_JIT", plan.install_steps[4].env)
        self.assertEqual(plan.install_steps[4].reason, plan.install_steps[3].reason)

    def test_build_native_runtime_plan_uses_linux_arm64_source_build_for_cu124_lane(self) -> None:
        venv_dir = Path("/tmp/native-plan-cu124")
        with (
            mock.patch.object(
                self.setup,
                "resolve_cuda_toolkit_for_label",
                return_value={
                    "desired_cuda_label": "cu124",
                    "toolkit_root": "/usr/local/cuda-12.4",
                    "include_dir": "/usr/local/cuda-12.4/include",
                    "nvcc_path": "/usr/local/cuda-12.4/bin/nvcc",
                    "nvcc_version": "12.4",
                    "status": "matched",
                    "reason": "ok",
                    "lib_dirs": ["/usr/local/cuda-12.4/lib64"],
                },
            ),
            mock.patch.dict(self.setup.os.environ, {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/usr/lib"}, clear=False),
        ):
            plan = self.setup.build_native_runtime_plan(
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=89,
                cuda_version=124,
                native_mode="install",
                venv_dir=venv_dir,
            )

        self.assertEqual(plan.host.lane, "linux-arm64-cu124")
        self.assertEqual(plan.install_strategy, "linux-arm64-source-build")
        self.assertNotIn("TORCH_CUDA_ARCH_LIST", plan.install_steps[0].env)
        self.assertEqual(plan.install_steps[0].env["FORCE_CUDA"], "1")
        self._assert_cuda_env_targets_selected_toolkit(plan.install_steps[0].env, toolkit_root="/usr/local/cuda-12.4")
        self._assert_native_path_order(
            plan.install_steps[0].env["PATH"],
            venv_dir=venv_dir,
            cuda_bin="/usr/local/cuda-12.4/bin",
            existing_path="/usr/bin",
        )
        self.assertTrue(plan.install_steps[0].env["LD_LIBRARY_PATH"].startswith("/usr/local/cuda-12.4/lib64:"))
        self.assertEqual(plan.install_steps[1].env["TORCH_CUDA_ARCH_LIST"], "8.9")
        self.assertEqual(plan.install_steps[3].env["CUMM_CUDA_ARCH_LIST"], "8.9")
        self.assertEqual(plan.install_steps[3].env["CUMM_DISABLE_JIT"], "1")
        self.assertIsNone(plan.install_steps[4].reason)
        self.assertNotIn("TORCH_CUDA_ARCH_LIST", plan.install_steps[3].env)
        self.assertEqual(plan.install_steps[4].env["SPCONV_DISABLE_JIT"], "1")
        self.assertNotIn("CUMM_DISABLE_JIT", plan.install_steps[4].env)
        self.assertEqual(plan.install_steps[4].env["CUMM_CUDA_ARCH_LIST"], "8.9")
        self.assertIsNone(plan.install_steps[3].reason)

    def test_build_native_runtime_plan_omits_unsupported_arch_overrides_for_sm121(self) -> None:
        venv_dir = Path("/tmp/native-plan-sm121")
        with (
            mock.patch.object(
                self.setup,
                "resolve_cuda_toolkit_for_label",
                return_value={
                    "desired_cuda_label": "cu128",
                    "toolkit_root": "/usr/local/cuda-12.8",
                    "include_dir": "/usr/local/cuda-12.8/include",
                    "nvcc_path": "/usr/local/cuda-12.8/bin/nvcc",
                    "nvcc_version": "12.8",
                    "status": "matched",
                    "reason": "ok",
                    "lib_dirs": ["/usr/local/cuda-12.8/lib64"],
                },
            ),
            mock.patch.dict(self.setup.os.environ, {"PATH": "/usr/bin"}, clear=False),
        ):
            plan = self.setup.build_native_runtime_plan(
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=121,
                cuda_version=128,
                native_mode="install",
                venv_dir=venv_dir,
            )

        self.assertEqual(plan.install_steps[1].package, "torch_scatter")
        self._assert_cuda_env_targets_selected_toolkit(plan.install_steps[1].env, toolkit_root="/usr/local/cuda-12.8")
        for step in plan.install_steps:
            self._assert_native_path_order(
                step.env["PATH"],
                venv_dir=venv_dir,
                cuda_bin="/usr/local/cuda-12.8/bin",
                existing_path="/usr/bin",
            )
        self.assertNotIn("TORCH_CUDA_ARCH_LIST", plan.install_steps[1].env)
        self.assertNotIn("CUMM_CUDA_ARCH_LIST", plan.install_steps[2].env)
        self.assertEqual(plan.install_steps[3].env["CUMM_DISABLE_JIT"], "1")
        self.assertIn("gpu_sm=121 (12.1)", plan.install_steps[3].reason)
        self.assertNotIn("TORCH_CUDA_ARCH_LIST", plan.install_steps[3].env)
        self.assertNotIn("CUMM_CUDA_ARCH_LIST", plan.install_steps[4].env)
        self.assertEqual(plan.install_steps[4].env["SPCONV_DISABLE_JIT"], "1")
        self.assertNotIn("CUMM_DISABLE_JIT", plan.install_steps[4].env)
        self.assertEqual(plan.install_steps[4].reason, plan.install_steps[3].reason)

    def test_cumm_cuda_arch_list_value_preserves_supported_sm89_and_omits_sm100_sm121(self) -> None:
        self.assertEqual(self.setup._cumm_cuda_arch_list_value(89), "8.9")
        self.assertIsNone(self.setup._cumm_cuda_arch_list_value(100))
        self.assertIsNone(self.setup._cumm_cuda_arch_list_value(121))

    def test_native_install_step_envs_prepend_venv_bin_before_cuda_and_existing_path(self) -> None:
        venv_dir = Path("/tmp/native-env-order")
        toolkit = {
            "toolkit_root": "/usr/local/cuda-12.8",
            "lib_dirs": ["/usr/local/cuda-12.8/lib64"],
        }

        with mock.patch.dict(self.setup.os.environ, {"PATH": "/usr/bin:/bin", "LD_LIBRARY_PATH": "/usr/lib"}, clear=False):
            steps = self.setup._linux_arm64_native_install_steps(gpu_sm=121, toolkit=toolkit, venv_dir=venv_dir)

        self.assertEqual([step.package for step in steps], [
            "native-build-prerequisites",
            "torch_scatter",
            "torch_cluster",
            "cumm",
            "spconv",
        ])
        for step in steps:
            self._assert_native_path_order(
                step.env["PATH"],
                venv_dir=venv_dir,
                cuda_bin="/usr/local/cuda-12.8/bin",
                existing_path="/usr/bin:/bin",
            )
            self._assert_cuda_env_targets_selected_toolkit(step.env, toolkit_root="/usr/local/cuda-12.8")
        self.assertNotIn("TORCH_CUDA_ARCH_LIST", steps[1].env)
        self.assertNotIn("CUMM_CUDA_ARCH_LIST", steps[2].env)
        self.assertEqual(steps[3].env["CUMM_DISABLE_JIT"], "1")
        self.assertNotIn("CUMM_CUDA_ARCH_LIST", steps[4].env)
        self.assertEqual(steps[4].env["SPCONV_DISABLE_JIT"], "1")

    def test_native_common_build_env_strips_default_cuda13_include_from_include_envs(self) -> None:
        toolkit = {
            "toolkit_root": "/usr/local/cuda-12.8",
            "include_dir": "/usr/local/cuda-12.8/include",
            "nvcc_path": "/usr/local/cuda-12.8/bin/nvcc",
            "lib_dirs": ["/usr/local/cuda-12.8/lib64"],
        }

        with mock.patch.dict(
            self.setup.os.environ,
            {
                "PATH": "/usr/bin",
                "CPATH": "/usr/local/cuda/include:/opt/include",
                "C_INCLUDE_PATH": "/usr/local/cuda/include:/opt/c/include",
                "CPLUS_INCLUDE_PATH": "/usr/local/cuda/include:/opt/cpp/include",
                "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib",
                "LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib64",
            },
            clear=False,
        ):
            env = self.setup._native_common_build_env(toolkit=toolkit, venv_dir=Path("/tmp/native-env"))

        self._assert_cuda_env_targets_selected_toolkit(env, toolkit_root="/usr/local/cuda-12.8")
        self.assertEqual(env["CPATH"], "/usr/local/cuda-12.8/include:/opt/include")
        self.assertEqual(env["C_INCLUDE_PATH"], "/usr/local/cuda-12.8/include:/opt/c/include")
        self.assertEqual(env["CPLUS_INCLUDE_PATH"], "/usr/local/cuda-12.8/include:/opt/cpp/include")
        self.assertEqual(env["LD_LIBRARY_PATH"], "/usr/local/cuda-12.8/lib64:/usr/lib")
        self.assertEqual(env["LIBRARY_PATH"], "/usr/local/cuda-12.8/lib64:/usr/lib64")

    def test_canonicalize_cumm_cuda_resolver_rewrites_known_linux_fragment(self) -> None:
        patched, status, reason = self.setup._canonicalize_cumm_cuda_resolver(
            self.setup.CUMM_CUDA_RESOLVER_LINUX_FRAGMENT
        )

        self.assertEqual(status, "patched")
        self.assertIsNone(reason)
        self.assertIn(self.setup.CUMM_CUDA_RESOLVER_PATCH_SENTINEL, patched)
        self.assertIn('cuda_root / "targets/sbsa-linux/include"', patched)
        self.assertNotIn('linux_cuda_root = Path("/usr/local/cuda")\n            include = linux_cuda_root / f"include"', patched.split(self.setup.CUMM_CUDA_RESOLVER_PATCH_SENTINEL, 1)[0])

    def test_patch_cumm_cuda_resolver_rewrites_managed_cumm_common_without_symlink_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, site_packages, _, _, _ = self._create_fake_cumm_layout(Path(temp_dir))
            common_path = self._write_fake_cumm_common(site_packages)
            with mock.patch.object(Path, "symlink_to") as symlink_mock:
                result = self.setup.patch_cumm_cuda_resolver(
                    venv_dir,
                    toolkit_root="/usr/local/cuda-12.8",
                    include_dir="/usr/local/cuda-12.8/include",
                    nvcc_path="/usr/local/cuda-12.8/bin/nvcc",
                )
            patched_source = common_path.read_text(encoding="utf-8")

        symlink_mock.assert_not_called()
        self.assertEqual(result["status"], "patched")
        self.assertEqual(result["action"], "rewrite")
        self.assertTrue(result["no_global_cuda_mutation"])
        self.assertEqual(result["selected_toolkit_root"], "/usr/local/cuda-12.8")
        self.assertEqual(result["selected_include_dir"], "/usr/local/cuda-12.8/include")
        self.assertEqual(result["selected_nvcc_path"], "/usr/local/cuda-12.8/bin/nvcc")
        self.assertIn(self.setup.CUMM_CUDA_RESOLVER_PATCH_SENTINEL, patched_source)
        self.assertIn('cuda_root / "targets/sbsa-linux/include"', patched_source)

    def test_patch_cumm_cuda_resolver_fails_closed_on_unknown_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, site_packages, _, _, _ = self._create_fake_cumm_layout(Path(temp_dir))
            self._write_fake_cumm_common(site_packages, source="unknown\n")

            result = self.setup.patch_cumm_cuda_resolver(
                venv_dir,
                toolkit_root="/usr/local/cuda-12.8",
                include_dir="/usr/local/cuda-12.8/include",
                nvcc_path="/usr/local/cuda-12.8/bin/nvcc",
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "unknown_cumm_cuda_resolver_source")

    def test_build_native_runtime_plan_supports_linux_arm64_with_detected_gpu_threshold(self) -> None:
        details = {
            "gpu_sm": 121,
            "cuda_version": 130,
            "gpu_sm_detail": {"field": "gpu_sm", "value": 121, "source": "nvidia-smi"},
            "cuda_version_detail": {"field": "cuda_version", "value": 130, "source": "nvidia-smi"},
            "nvidia_smi": {"status": "detected", "errors": [], "commands": {}},
        }

        plan = self.setup.build_native_runtime_plan(
            system_name="Linux",
            machine="aarch64",
            py_tag="cp311",
            gpu_sm=self.setup._resolve_gpu_sm({"gpu_sm": 0}, details),
            cuda_version=self.setup._resolve_cuda_version({"cuda_version": 0}, details),
            native_mode="auto",
        )

        self.assertEqual(plan.host.state, "supported")
        self.assertEqual(plan.host.lane, "linux-arm64-cu128")

    def test_build_native_runtime_plan_blocks_when_matching_toolkit_missing(self) -> None:
        with mock.patch.object(
            self.setup,
            "resolve_cuda_toolkit_for_label",
            return_value={
                "desired_cuda_label": "cu128",
                "toolkit_root": None,
                "nvcc_version": None,
                "status": "blocked",
                "reason": "CUDA 12.8 toolkit missing",
                "lib_dirs": [],
            },
        ):
            plan = self.setup.build_native_runtime_plan(
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                native_mode="install",
            )

        self.assertEqual(plan.status, "blocked")
        self.assertFalse(plan.automatic_install_supported)
        self.assertEqual(plan.toolkit_status, "blocked")
        self.assertEqual(plan.install_steps[0].status, "blocked")
        self.assertEqual(plan.next_action, "CUDA 12.8 toolkit missing")

    def test_verify_imports_do_not_expect_top_level_sonata_package(self) -> None:
        self.assertNotIn("sonata", self.setup.VERIFY_IMPORTS)
        self.assertIn("addict", self.setup.VERIFY_IMPORTS)
        self.assertIn("omegaconf", self.setup.VERIFY_IMPORTS)
        self.assertIn("skimage", self.setup.VERIFY_IMPORTS)
        self.assertIn("diffusers", self.setup.VERIFY_IMPORTS)
        self.assertIn("einops", self.setup.VERIFY_IMPORTS)
        self.assertIn("pymeshlab", self.setup.VERIFY_IMPORTS)

    def test_runtime_packages_include_scikit_image_for_x_part_import_smoke(self) -> None:
        self.assertIn("scikit-image", self.setup.RUNTIME_PACKAGES)
        self.assertIn("diffusers", self.setup.RUNTIME_PACKAGES)
        self.assertIn("einops", self.setup.RUNTIME_PACKAGES)
        self.assertIn("pymeshlab", self.setup.RUNTIME_PACKAGES)

    def test_build_readiness_report_marks_missing_venv_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            with (
                mock.patch.object(self.setup, "python_tag", return_value="cp312"),
                mock.patch.object(
                    self.setup,
                    "verify_runtime",
                    return_value={"ready": False, "missing": ["torch"], "python_exe": "/usr/bin/python3"},
                ),
                mock.patch.object(
                    self.setup,
                    "build_adapter_readiness",
                    return_value=mock.Mock(
                        ready=False,
                        to_dict=lambda: {
                            "ready": False,
                            "status": "blocked",
                            "components": {
                                "import_smoke": {"ready": False, "status": "missing", "missing_modules": []}
                            },
                        },
                        components={"import_smoke": {"ready": False, "status": "missing", "missing_modules": []}},
                    ),
                ),
            ):
                report = self.setup.build_readiness_report(project_root=project_root)

        self.assertEqual(report["status"], "pending")
        self.assertFalse(report["venv"]["exists"])
        self.assertFalse(report["execution_ready"])
        self.assertIn("gpu_detection", report)
        self.assertIn("base_runtime", report)
        self.assertIn("native_runtime", report)
        self.assertIn("phases", report)
        self.assertFalse(report["upstream_runtime"]["ready"])
        self.assertEqual(report["runtime_adapter"]["status"], "blocked")
        self.assertEqual(report["native_runtime"]["mode"], "auto")
        self.assertEqual(report["native_dependency_policy"]["strategy"], "unsupported")
        self.assertEqual(report["native_dependency_blockers"]["status"], "pending")

    def test_build_readiness_report_includes_gpu_detection_details(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            with (
                mock.patch.object(self.setup, "python_tag", return_value="cp312"),
                mock.patch.object(
                    self.setup,
                    "verify_runtime",
                    return_value={"ready": False, "missing": ["torch"], "python_exe": "/usr/bin/python3"},
                ),
                mock.patch.object(
                    self.setup,
                    "build_adapter_readiness",
                    return_value=mock.Mock(
                        ready=False,
                        to_dict=lambda: {"ready": False, "status": "blocked", "components": {}},
                        components={"import_smoke": {"ready": False, "status": "missing", "missing_modules": []}},
                    ),
                ),
            ):
                report = self.setup.build_readiness_report(
                    project_root=project_root,
                    gpu_sm=121,
                    cuda_version=130,
                    gpu_runtime_details={
                        "gpu_sm": 121,
                        "cuda_version": 130,
                        "gpu_sm_detail": {"field": "gpu_sm", "value": 121, "source": "nvidia-smi"},
                        "cuda_version_detail": {"field": "cuda_version", "value": 130, "source": "nvidia-smi"},
                        "nvidia_smi": {
                            "status": "detected",
                            "selected_compute_capability": "12.1",
                            "compute_capabilities": ["12.1"],
                            "cuda_version_text": "13.0",
                            "errors": [],
                            "commands": {},
                        },
                    },
                )

        self.assertEqual(report["gpu_detection"]["gpu_sm"], 121)
        self.assertEqual(report["gpu_detection"]["cuda_version"], 130)
        self.assertEqual(report["gpu_detection"]["gpu_sm_detail"]["source"], "nvidia-smi")

    def test_build_readiness_report_separates_native_runtime_from_missing_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            managed_python = project_root / "venv" / "bin" / "python"
            managed_python.parent.mkdir(parents=True, exist_ok=True)
            managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            runtime_root = self.setup.runtime_source_root(project_root)
            for relative_path in self.setup.UPSTREAM_REQUIRED_PATHS:
                target = runtime_root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("ok\n", encoding="utf-8")

            runtime_adapter = mock.Mock(
                ready=False,
                status="blocked",
                to_dict=lambda: {
                    "ready": False,
                    "status": "blocked",
                    "components": {
                        "weights": {"ready": False, "status": "missing"},
                        "import_smoke": {"ready": True, "status": "ready", "missing_modules": [], "native_blockers": []},
                    },
                },
                components={
                    "weights": {"ready": False, "status": "missing"},
                    "import_smoke": {"ready": True, "status": "ready", "missing_modules": [], "native_blockers": []},
                },
            )

            with (
                mock.patch.object(self.setup, "python_tag", return_value="cp312"),
                mock.patch.object(
                    self.setup,
                    "verify_runtime",
                    return_value={
                        "ready": True,
                        "missing": [],
                        "python_exe": str(managed_python),
                        "python_version": "3.12.3",
                        "python_abi": "cp312",
                        "torch_imported": True,
                        "cuda_available": True,
                    },
                ),
                mock.patch.object(self.setup, "build_adapter_readiness", return_value=runtime_adapter),
                mock.patch.object(
                    self.setup,
                    "run_native_runtime_phase",
                    return_value={
                        "mode": "install",
                        "ready": True,
                        "status": "ready",
                        "message": "Native runtime imports are ready.",
                        "next_action": "Download weights before inference.",
                        "install_strategy": "linux-arm64-source-build",
                        "host": {"system": "Linux", "machine": "aarch64"},
                        "plan": {"status": "planned", "install_strategy": "linux-arm64-source-build", "required_imports": ["spconv.core_cc", "spconv.pytorch"]},
                        "probe": {"blocked_imports": [], "missing_modules": [], "message": "ok"},
                        "install_results": [],
                    },
                ),
                mock.patch.object(
                    self.setup,
                    "describe_upstream_runtime_source",
                    return_value={"ready": True, "status": "prepared", "source": str(runtime_root)},
                ),
                mock.patch.object(
                    self.setup.Hunyuan3DPartGenerator,
                    "readiness_status",
                    return_value={"execution_ready": False, "weights": {"status": "missing"}, "runtime_adapter": {"status": "blocked"}},
                ),
            ):
                report = self.setup.build_readiness_report(project_root=project_root)

        self.assertTrue(report["native_runtime_ready"])
        self.assertFalse(report["weights_ready"])
        self.assertFalse(report["adapter_ready"])
        self.assertFalse(report["runtime_ready"])
        self.assertFalse(report["execution_ready"])
        self.assertEqual(report["native_runtime"]["status"], "ready")
        self.assertEqual(report["weights"]["status"], "missing")
        self.assertEqual(report["phases"]["weights"]["status"], "missing")
        self.assertEqual(report["phases"]["native_runtime"]["status"], "ready")
        self.assertEqual(report["support"]["source"], "managed_python_torch")
        self.assertTrue(report["host"]["cuda_visible"])
        self.assertEqual(report["host"]["cuda_visible_source"], "managed_python_torch")
        self.assertNotIn("Runtime imports failed after managed setup.", report["blocked_reasons"])
        self.assertTrue(any("Missing required model weights:" in reason for reason in report["blocked_reasons"]))

    def test_weight_status_uses_modly_central_model_cache_for_extensions_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            modly_root = Path(temp_dir) / "Modly"
            project_root = modly_root / "extensions" / "hunyuan3d-part"
            project_root.mkdir(parents=True, exist_ok=True)
            weight_path = modly_root / "models" / "hunyuan3d-part" / "p3sam" / "p3sam" / "p3sam.safetensors"
            weight_path.parent.mkdir(parents=True, exist_ok=True)
            weight_path.write_bytes(b"weights")

            status = self.setup._weight_status(project_root)

        self.assertTrue(status["ready"])
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["model_root"], str(modly_root / "models" / "hunyuan3d-part"))
        self.assertEqual(status["path"], str(weight_path))
        self.assertNotIn(str(project_root / ".runtime" / "models"), status["path"])

    def test_native_dependency_report_marks_spconv_and_torch_scatter_blockers(self) -> None:
        report = self.setup.build_native_dependency_report(
            native_runtime={
                "mode": "auto",
                "ready": False,
                "status": "blocked",
                "host": {"system": "Linux", "machine": "arm64", "lane": "linux-arm64-cu124"},
                "plan": {
                    "required_imports": ["spconv.pytorch", "torch_scatter", "torch_cluster"],
                    "related_packages": ["cumm"],
                    "status": "planned",
                    "install_strategy": "linux-arm64-source-build",
                    "automatic_install_supported": False,
                    "install_steps": [],
                    "message": "plan",
                    "next_action": "repair",
                },
                "probe": {
                    "blocked_imports": ["spconv.pytorch", "torch_scatter", "torch_cluster"],
                    "missing_modules": ["spconv", "spconv.pytorch", "torch_scatter", "torch_cluster"],
                },
                "message": "blocked",
                "next_action": "repair",
            }
        )

        self.assertEqual(report["native_dependency_policy"]["required_imports"], ["spconv.pytorch", "torch_scatter", "torch_cluster"])
        self.assertEqual(report["native_dependency_policy"]["related_native_packages"], ["cumm"])
        self.assertEqual(report["native_dependency_blockers"]["status"], "blocked")
        self.assertEqual(report["native_dependency_blockers"]["blocked_imports"], ["spconv.pytorch", "torch_scatter", "torch_cluster"])
        self.assertEqual(report["native_dependency_blockers"]["strategy"], "plan-and-probe")
        self.assertEqual(report["native_dependency_policy"]["install_strategy"], "linux-arm64-source-build")
        self.assertEqual(report["native_dependency_blockers"]["next_action"], "repair")

    def test_probe_native_runtime_reports_missing_native_modules(self) -> None:
        plan = self.setup.build_native_runtime_plan(
            system_name="Linux",
            machine="aarch64",
            py_tag="cp311",
            gpu_sm=100,
            cuda_version=128,
            native_mode="auto",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            managed_python = Path(temp_dir) / "python"
            managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            with mock.patch.object(self.setup.subprocess, "run") as run_mock:
                run_mock.return_value = mock.Mock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "spconv": {"ready": False, "status": "missing", "error": "No module named 'spconv'"},
                            "spconv.pytorch": {"ready": False, "status": "missing", "error": "No module named 'spconv'"},
                            "torch_scatter": {"ready": False, "status": "missing", "error": "No module named 'torch_scatter'"},
                            "torch_cluster": {"ready": False, "status": "missing", "error": "No module named 'torch_cluster'"},
                            "cumm": {"ready": False, "status": "missing", "error": "No module named 'cumm'"},
                        }
                    ),
                    stderr="",
                )
                probe = self.setup.probe_native_runtime(managed_python=managed_python, plan=plan)

        self.assertFalse(probe.ready)
        self.assertEqual(probe.status, "blocked")
        self.assertEqual(probe.blocked_imports, ("spconv.pytorch", "torch_scatter", "torch_cluster"))
        self.assertIn("spconv", probe.missing_modules)

    def test_probe_cumm_core_cc_reports_ready_when_import_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            managed_python = Path(temp_dir) / "python"
            managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            with mock.patch.object(self.setup.subprocess, "run") as run_mock:
                run_mock.return_value = mock.Mock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "package": "cumm-core-cc",
                            "module": "cumm.core_cc",
                            "ready": True,
                            "status": "ready",
                            "managed_python": str(managed_python),
                            "error": None,
                            "error_type": None,
                            "traceback_tail": [],
                            "next_action": "None. cumm.core_cc imports successfully.",
                        }
                    ),
                    stderr="",
                )
                result = self.setup.probe_cumm_core_cc(managed_python=managed_python)

        self.assertTrue(result["ready"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["module"], "cumm.core_cc")

    def test_probe_cumm_core_cc_reports_module_not_found_with_traceback_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            managed_python = Path(temp_dir) / "python"
            managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            with mock.patch.object(self.setup.subprocess, "run") as run_mock:
                run_mock.return_value = mock.Mock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "package": "cumm-core-cc",
                            "module": "cumm.core_cc",
                            "ready": False,
                            "status": "missing",
                            "managed_python": str(managed_python),
                            "error": "No module named 'cumm.core_cc'",
                            "error_type": "ModuleNotFoundError",
                            "traceback_tail": [
                                "Traceback (most recent call last):",
                                "  File \"<string>\", line 4, in <module>",
                                "ModuleNotFoundError: No module named 'cumm.core_cc'",
                            ],
                            "next_action": "Inspect the managed cumm install and ensure cumm.core_cc is built before attempting spconv.",
                        }
                    ),
                    stderr="",
                )
                result = self.setup.probe_cumm_core_cc(managed_python=managed_python)

        self.assertFalse(result["ready"])
        self.assertEqual(result["status"], "missing")
        self.assertEqual(result["error_type"], "ModuleNotFoundError")
        self.assertEqual(result["traceback_tail"][-1], "ModuleNotFoundError: No module named 'cumm.core_cc'")

    def test_probe_spconv_core_cc_reports_ready_when_import_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            managed_python = Path(temp_dir) / "python"
            managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            with mock.patch.object(self.setup.subprocess, "run") as run_mock:
                run_mock.return_value = mock.Mock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "package": "spconv-core-cc",
                            "module": "spconv.core_cc",
                            "ready": True,
                            "status": "ready",
                            "managed_python": str(managed_python),
                            "error": None,
                            "error_type": None,
                            "traceback_tail": [],
                            "next_action": "None. spconv.core_cc imports successfully.",
                        }
                    ),
                    stderr="",
                )
                result = self.setup.probe_spconv_core_cc(managed_python=managed_python)

        self.assertTrue(result["ready"])
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["module"], "spconv.core_cc")

    def test_probe_spconv_core_cc_reports_module_not_found_with_traceback_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            managed_python = Path(temp_dir) / "python"
            managed_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            with mock.patch.object(self.setup.subprocess, "run") as run_mock:
                run_mock.return_value = mock.Mock(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "package": "spconv-core-cc",
                            "module": "spconv.core_cc",
                            "ready": False,
                            "status": "missing",
                            "managed_python": str(managed_python),
                            "error": "No module named 'spconv.core_cc'",
                            "error_type": "ModuleNotFoundError",
                            "traceback_tail": [
                                "Traceback (most recent call last):",
                                "  File \"<string>\", line 4, in <module>",
                                "ModuleNotFoundError: No module named 'spconv.core_cc'",
                            ],
                            "next_action": "Inspect the managed spconv install and rebuild with SPCONV_DISABLE_JIT=1 so spconv.core_cc is produced during install.",
                        }
                    ),
                    stderr="",
                )
                result = self.setup.probe_spconv_core_cc(managed_python=managed_python)

        self.assertFalse(result["ready"])
        self.assertEqual(result["status"], "missing")
        self.assertEqual(result["error_type"], "ModuleNotFoundError")
        self.assertEqual(result["traceback_tail"][-1], "ModuleNotFoundError: No module named 'spconv.core_cc'")

    def test_inspect_cumm_tensorview_layout_reports_expected_include_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, site_packages, package_root, expected_include, _sentinel = self._create_fake_cumm_layout(Path(temp_dir))

            layout = self.setup.inspect_cumm_tensorview_layout(venv_dir)

        self.assertEqual(layout["status"], "blocked")
        self.assertEqual(layout["site_packages"], str(site_packages))
        self.assertEqual(layout["package_root"], str(package_root.resolve()))
        self.assertEqual(layout["expected_include_path"], str(expected_include.resolve()))
        self.assertEqual(layout["fallback_include_path"], str((package_root / "include").resolve()))

    def test_repair_cumm_tensorview_headers_is_idempotent_when_expected_path_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")

            result = self.setup.repair_cumm_tensorview_headers(venv_dir, extension_dir=Path(temp_dir))

        self.assertEqual(result["status"], "already_present")
        self.assertEqual(result["action"], "noop")
        self.assertEqual(result["expected_path"], str(expected_include.resolve()))

    def test_repair_cumm_tensorview_headers_copies_from_source_candidate_when_symlink_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            venv_dir, _site_packages, _package_root, expected_include, _sentinel = self._create_fake_cumm_layout(temp_root)
            source_root = temp_root / "cached-cumm"
            source_include = source_root / "include" / "tensorview"
            source_include.mkdir(parents=True, exist_ok=True)
            (source_include / "tensor.h").write_text("// header\n", encoding="utf-8")

            with mock.patch.object(Path, "symlink_to", side_effect=OSError("no symlink")):
                result = self.setup.repair_cumm_tensorview_headers(
                    venv_dir,
                    extension_dir=temp_root,
                    source_search_roots=(source_root,),
                    source_preparer=mock.Mock(),
                )

            repaired_header = expected_include / "tensorview" / "tensor.h"
            self.assertTrue(repaired_header.is_file())

        self.assertEqual(result["status"], "repaired")
        self.assertEqual(result["action"], "copy")
        self.assertEqual(result["source_path"], str((source_root / "include").resolve()))

    def test_repair_cumm_tensorview_headers_returns_structured_failure_when_source_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            venv_dir, _site_packages, _package_root, expected_include, _sentinel = self._create_fake_cumm_layout(temp_root)

            result = self.setup.repair_cumm_tensorview_headers(
                venv_dir,
                extension_dir=temp_root,
                source_search_roots=(temp_root / "missing-source",),
                source_preparer=mock.Mock(
                    return_value={
                        "status": "failed",
                        "action": "clone",
                        "reason": "headers not found",
                        "next_action": "clone cumm include tree",
                    }
                ),
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["action"], "clone")
        self.assertEqual(result["reason"], "headers not found")
        self.assertEqual(result["expected_path"], str(expected_include.resolve()))

    def test_patch_cumm_subint_header_applies_guarded_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            header_path = self._write_fake_cumm_subint_header(expected_include)

            result = self.setup.patch_cumm_subint_header(venv_dir)
            patched = header_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "patched")
        self.assertEqual(result["package"], "cumm-subint-header-patch")
        self.assertEqual(self._extract_fake_cumm_subint_body(patched), self.setup.CUMM_SUBINT_PATCH_BODY_REPLACEMENT)
        self.assertIn("operator T() const {", patched)
        self._assert_no_custom_subint_comparison_operators(patched)
        self.assertNotIn("T value() const {", patched)
        self.assertNotIn("bool operator>(integer_subbyte const &rhs) const { return !(rhs < *this); }", patched)
        self.assertEqual(patched.count("template <int Bits, bool Signed = true> struct integer_subbyte {"), 1)
        self.assertEqual(patched.count("} // namespace tv"), 1)

    def test_patch_cumm_subint_header_accepts_original_body_with_trailing_newline_before_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            header_path = self._write_fake_cumm_subint_header(
                expected_include,
                source=self.setup.CUMM_SUBINT_PATCH_BODY_ORIGINAL + "\n",
            )

            result = self.setup.patch_cumm_subint_header(venv_dir)
            patched = header_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "patched")
        self.assertEqual(self._extract_fake_cumm_subint_body(patched), self.setup.CUMM_SUBINT_PATCH_BODY_REPLACEMENT)

    def test_patch_cumm_subint_header_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            self._write_fake_cumm_subint_header(expected_include)

            first = self.setup.patch_cumm_subint_header(venv_dir)
            second = self.setup.patch_cumm_subint_header(venv_dir)

        self.assertEqual(first["status"], "patched")
        self.assertEqual(second["status"], "already_patched")
        self.assertEqual(second["action"], "noop")

    def test_patch_cumm_subint_header_treats_patched_body_with_trailing_newline_as_already_patched(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            header_path = self._write_fake_cumm_subint_header(
                expected_include,
                source=self.setup.CUMM_SUBINT_PATCH_BODY_REPLACEMENT + "\n",
            )
            before = header_path.read_text(encoding="utf-8")

            result = self.setup.patch_cumm_subint_header(venv_dir)
            after = header_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "already_patched")
        self.assertEqual(result["action"], "noop")
        self.assertEqual(before, after)

    def test_patch_cumm_subint_header_replaces_complete_integer_subbyte_struct(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            header_path = self._write_fake_cumm_subint_header(expected_include)

            self.setup.patch_cumm_subint_header(venv_dir)
            patched = header_path.read_text(encoding="utf-8")

        struct_block = self._extract_fake_cumm_subint_body(patched)
        self.assertIn("operator T() const {", struct_block)
        self._assert_no_custom_subint_comparison_operators(struct_block)
        self.assertNotIn("T value() const {", struct_block)
        self.assertNotIn("operator>(integer_subbyte const &rhs)", struct_block)
        self.assertEqual(struct_block.count("template <int Bits, bool Signed = true> struct integer_subbyte {"), 1)
        self.assertEqual(struct_block.count("using uint1b_t = integer_subbyte<1, false>;"), 1)
        self.assertEqual(struct_block.count("template <> struct sizeof_bits<uint4b_t> { static int const value = 4; };"), 1)

    def test_patch_cumm_subint_header_rewrites_legacy_replacement_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            header_path = self._write_fake_cumm_subint_header(
                expected_include,
                source=self.setup.CUMM_SUBINT_PATCH_BODY_REPLACEMENT_LEGACY,
            )

            result = self.setup.patch_cumm_subint_header(venv_dir)
            patched = header_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "patched")
        self.assertEqual(self._extract_fake_cumm_subint_body(patched), self.setup.CUMM_SUBINT_PATCH_BODY_REPLACEMENT)

    def test_patch_cumm_subint_header_repairs_known_malformed_patched_body(self) -> None:
        malformed_body = self.setup.CUMM_SUBINT_PATCH_BODY_REPLACEMENT.replace(
            "\n\n/// 1-bit Unsigned integer type\n",
            "\n\nTV_HOST_DEVICE_INLINE\nbool operator<(self_type const &rhs) const { return value() < rhs.value(); }\n\nTV_HOST_DEVICE_INLINE\nbool operator>=(self_type const &rhs) const { return value() >= rhs.value(); }\n\nTV_HOST_DEVICE_INLINE\nbool operator>(self_type const &rhs) const { return value() > rhs.value(); }\n\n/// 1-bit Unsigned integer type\n",
            1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            header_path = self._write_fake_cumm_subint_header(expected_include, source=malformed_body)

            result = self.setup.patch_cumm_subint_header(venv_dir)
            patched = header_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "patched")
        self.assertEqual(self._extract_fake_cumm_subint_body(patched), self.setup.CUMM_SUBINT_PATCH_BODY_REPLACEMENT)
        self._assert_no_custom_subint_comparison_operators(patched)

    def test_cumm_subint_replacement_body_keeps_template_alias_and_sizeof_order(self) -> None:
        body = self.setup.CUMM_SUBINT_PATCH_BODY_REPLACEMENT
        template_index = body.index("template <int Bits, bool Signed = true> struct integer_subbyte {")
        alias_index = body.index("using int4b_t = integer_subbyte<4, true>;")
        sizeof_int4_index = body.index("template <> struct sizeof_bits<int4b_t> { static int const value = 4; };")
        sizeof_uint4_index = body.index("template <> struct sizeof_bits<uint4b_t> { static int const value = 4; };")

        self.assertLess(template_index, alias_index)
        self.assertLess(alias_index, sizeof_int4_index)
        self.assertLess(sizeof_int4_index, sizeof_uint4_index)

    def test_cumm_subint_replacement_body_has_balanced_braces_and_no_trailing_operator_fragments(self) -> None:
        body = self.setup.CUMM_SUBINT_PATCH_BODY_REPLACEMENT

        self._assert_balanced_braces(body)
        self._assert_no_free_floating_subint_operators(body)

    def test_cumm_subint_replacement_body_removes_custom_comparison_operators(self) -> None:
        body = self.setup.CUMM_SUBINT_PATCH_BODY_REPLACEMENT

        self.assertIn("operator T() const {", body)
        self._assert_no_custom_subint_comparison_operators(body)

    def test_patch_cumm_subint_header_fails_closed_on_unknown_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            header_path = self._write_fake_cumm_subint_header(expected_include, source="struct unexpected {};\n")
            before = header_path.read_text(encoding="utf-8")

            result = self.setup.patch_cumm_subint_header(venv_dir)
            after = header_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "unknown_subint_header_source")
        self.assertEqual(before, after)

    def test_patch_cumm_half_header_inserts_unconditional_undef_block_including_copysign(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            header_path = self._write_fake_cumm_half_header(expected_include)

            result = self.setup.patch_cumm_half_header(venv_dir)
            patched = header_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "patched")
        self.assertEqual(result["package"], "cumm-half-header-patch")
        self.assertEqual(result["action"], "rewrite")
        self.assertEqual(patched, self._patched_half_header_source())
        self.assertIn("#ifdef copysign\n#undef copysign\n#endif\n", patched)
        self.assertEqual(patched.count("#ifdef copysign\n#undef copysign\n#endif\n"), 1)
        self.assertIn("  TV_HOST_DEVICE_INLINE\n  static half_t convert(float const &flt) {\n", patched)
        self.assertEqual(patched.count("  TV_HOST_DEVICE_INLINE\n  static half_t convert(float const &flt) {\n"), 1)
        self.assertIn("  TV_HOST_DEVICE_INLINE\n  static float convert(half_t const &x) {\n", patched)
        self.assertEqual(patched.count("  TV_HOST_DEVICE_INLINE\n  static float convert(half_t const &x) {\n"), 1)
        self.assertNotIn("__device__ __noinline__", patched)
        self.assertNotIn("static TV_HOST_DEVICE_INLINE half_t convert(float const &flt) {", patched)
        self.assertNotIn("TV_HOST_DEVICE_INLINE\n      static half_t\n      convert(float const &flt)", patched)
        self.assertNotIn("static TV_HOST_DEVICE_INLINE float convert(half_t const &x) {", patched)
        self.assertNotIn("TV_HOST_DEVICE_INLINE\n      static float\n      convert(half_t const &x)", patched)
        self.assertIn("static TV_HOST_DEVICE_INLINE\nbool signbit(tv::half_t const &h)", patched)
        self.assertIn("static TV_HOST_DEVICE_INLINE\ntv::half_t copysign(tv::half_t const &a, tv::half_t const &b)", patched)

    def test_patch_cumm_half_header_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            self._write_fake_cumm_half_header(expected_include, source=self._patched_half_header_source())

            first = self.setup.patch_cumm_half_header(venv_dir)
            second = self.setup.patch_cumm_half_header(venv_dir)

        self.assertEqual(first["status"], "already_patched")
        self.assertEqual(second["status"], "already_patched")
        self.assertEqual(first["action"], "noop")
        self.assertEqual(second["action"], "noop")

    def test_canonicalize_cumm_half_header_canonicalizes_live_like_shape(self) -> None:
        patched, status, reason = self.setup._canonicalize_cumm_half_header(self._live_like_half_header_source())

        self.assertEqual(status, "patched")
        self.assertIsNone(reason)
        self.assertEqual(patched, self._patched_half_header_source())

    def test_canonicalize_cumm_half_header_is_idempotent_for_live_target(self) -> None:
        patched = self._patched_half_header_source()

        second, status, reason = self.setup._canonicalize_cumm_half_header(patched)

        self.assertEqual(status, "already_patched")
        self.assertIsNone(reason)
        self.assertEqual(second, patched)

    def test_canonicalize_cumm_half_header_repairs_latest_live_member_shape_without_touching_class_scope(self) -> None:
        patched, status, reason = self.setup._canonicalize_cumm_half_header(self._live_like_half_header_source())

        self.assertEqual(status, "patched")
        self.assertIsNone(reason)
        self.assertIn("  TV_HOST_DEVICE_INLINE\n  static half_t convert(float const &flt) {\n", patched)
        self.assertIn("  TV_HOST_DEVICE_INLINE\n  static float convert(half_t const &x) {\n", patched)
        self.assertNotIn("TV_HOST_DEVICE_INLINE\n      static half_t\n      convert(float const &flt)", patched)
        self.assertNotIn("TV_HOST_DEVICE_INLINE\n      static float\n      convert(half_t const &x)", patched)
        self.assertNotIn("__device__ __noinline__", patched)
        self.assertIn("static TV_HOST_DEVICE_INLINE\ntv::half_t copysign(tv::half_t const &a, tv::half_t const &b)", patched)

    def test_canonicalize_cumm_half_header_repairs_malformed_previous_target_to_safe_convert_prologue(self) -> None:
        patched, status, reason = self.setup._canonicalize_cumm_half_header(self._malformed_half_header_source())

        self.assertEqual(status, "patched")
        self.assertIsNone(reason)
        self.assertEqual(patched, self._patched_half_header_source())
        self.assertIn("  TV_HOST_DEVICE_INLINE\n  static half_t convert(float const &flt) {\n", patched)
        self.assertIn("  TV_HOST_DEVICE_INLINE\n  static float convert(half_t const &x) {\n", patched)
        self.assertNotIn("__device__ __noinline__", patched)

    def test_patched_half_header_source_keeps_helper_fixes_and_safe_convert_prologue(self) -> None:
        patched = self._patched_half_header_source()

        self.assertIn("  TV_HOST_DEVICE_INLINE\n  static half_t convert(float const &flt) {\n", patched)
        self.assertEqual(patched.count("  TV_HOST_DEVICE_INLINE\n  static half_t convert(float const &flt) {\n"), 1)
        self.assertIn("  TV_HOST_DEVICE_INLINE\n  static float convert(half_t const &x) {\n", patched)
        self.assertEqual(patched.count("  TV_HOST_DEVICE_INLINE\n  static float convert(half_t const &x) {\n"), 1)
        self.assertNotIn("__device__ __noinline__", patched)
        self.assertIn("static TV_HOST_DEVICE_INLINE\ntv::half_t copysign(tv::half_t const &a, tv::half_t const &b)", patched)

    def test_patch_cumm_bfloat16_header_canonicalizes_live_like_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            header_path = self._write_fake_cumm_bfloat16_header(expected_include)

            result = self.setup.patch_cumm_bfloat16_header(venv_dir)
            patched = header_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "patched")
        self.assertEqual(result["package"], "cumm-bfloat16-header-patch")
        self.assertEqual(patched, self._patched_bfloat16_header_source())
        self.assertIn("#ifdef signbit\n#undef signbit\n#endif\n", patched)
        self.assertIn("#ifdef sqrt\n#undef sqrt\n#endif\n", patched)
        self.assertNotIn("bool signbit(tv::bfloat16_t const &h)", patched)
        self.assertIn("static TV_HOST_DEVICE_INLINE\ntv::bfloat16_t copysign(tv::bfloat16_t const &a, tv::bfloat16_t const &b)", patched)

    def test_patch_cumm_bfloat16_header_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            self._write_fake_cumm_bfloat16_header(expected_include, source=self._patched_bfloat16_header_source())

            first = self.setup.patch_cumm_bfloat16_header(venv_dir)
            second = self.setup.patch_cumm_bfloat16_header(venv_dir)

        self.assertEqual(first["status"], "already_patched")
        self.assertEqual(second["status"], "already_patched")
        self.assertEqual(first["action"], "noop")
        self.assertEqual(second["action"], "noop")

    def test_canonicalize_cumm_bfloat16_header_drops_conflicting_namespace_signbit_helper(self) -> None:
        patched, status, reason = self.setup._canonicalize_cumm_bfloat16_header(self._live_like_bfloat16_header_source())

        self.assertEqual(status, "patched")
        self.assertIsNone(reason)
        self.assertNotIn("bool signbit(tv::bfloat16_t const &h)", patched)
        self.assertIn("static TV_HOST_DEVICE_INLINE\ntv::bfloat16_t copysign(tv::bfloat16_t const &a, tv::bfloat16_t const &b)", patched)

    def test_patch_cumm_bfloat16_header_fails_closed_on_unknown_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            header_path = self._write_fake_cumm_bfloat16_header(expected_include, source="#pragma once\nnamespace tv {}\n")
            before = header_path.read_text(encoding="utf-8")

            result = self.setup.patch_cumm_bfloat16_header(venv_dir)
            after = header_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "unknown_bfloat16_header_source")
        self.assertEqual(before, after)

    def test_patch_cumm_half_header_fails_closed_on_unknown_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            header_path = self._write_fake_cumm_half_header(expected_include, source="#pragma once\nnamespace tv {}\n")
            before = header_path.read_text(encoding="utf-8")

            result = self.setup.patch_cumm_half_header(venv_dir)
            after = header_path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "unknown_half_header_source")
        self.assertEqual(before, after)

    def test_patch_cumm_dtype_headers_rewrites_exact_canonical_snapshot_to_local_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            tf32_header = self._write_fake_cumm_tf32_header(expected_include, source=self._canonical_tf32_header_source())
            float8_header = self._write_fake_cumm_float8_header(expected_include, source=self._canonical_float8_header_source())

            result = self.setup.patch_cumm_dtype_headers(venv_dir)
            tf32_patched = tf32_header.read_text(encoding="utf-8")
            float8_patched = float8_header.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "patched")
        self.assertEqual(result["package"], "cumm-dtype-header-patch")
        self.assertEqual(result["action"], "rewrite")
        self.assertEqual(tf32_patched, self._local_target_tf32_header_source())
        self.assertEqual(float8_patched, self._local_target_float8_header_source())

    def test_patch_cumm_dtype_headers_rewrites_unguarded_variants_to_guarded_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            tf32_header = self._write_fake_cumm_tf32_header(expected_include)
            float8_header = self._write_fake_cumm_float8_header(expected_include)

            result = self.setup.patch_cumm_dtype_headers(venv_dir)
            tf32_patched = tf32_header.read_text(encoding="utf-8")
            float8_patched = float8_header.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "patched")
        self.assertEqual(tf32_patched, self._local_target_tf32_header_source())
        self.assertEqual(float8_patched, self._local_target_float8_header_source())

    def test_patch_cumm_dtype_headers_rewrites_malformed_variants_to_guarded_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            tf32_header = self._write_fake_cumm_tf32_malformed_patched_header(expected_include)
            float8_header = self._write_fake_cumm_float8_malformed_patched_header(expected_include)

            result = self.setup.patch_cumm_dtype_headers(venv_dir)
            tf32_patched = tf32_header.read_text(encoding="utf-8")
            float8_patched = float8_header.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "patched")
        self.assertEqual(tf32_patched, self._local_target_tf32_header_source())
        self.assertEqual(float8_patched, self._local_target_float8_header_source())

    def test_patch_cumm_dtype_headers_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            self._write_fake_cumm_tf32_header(expected_include, source=self._local_target_tf32_header_source())
            self._write_fake_cumm_float8_header(expected_include, source=self._local_target_float8_header_source())

            first = self.setup.patch_cumm_dtype_headers(venv_dir)
            second = self.setup.patch_cumm_dtype_headers(venv_dir)

        self.assertEqual(first["status"], "already_patched")
        self.assertEqual(second["status"], "already_patched")
        self.assertEqual(first["action"], "noop")
        self.assertEqual(second["action"], "noop")

    def test_local_target_tf32_header_uses_internal_linkage_for_math_helpers(self) -> None:
        target = self._local_target_tf32_header_source()

        self.assertIn("static TV_HOST_DEVICE_INLINE\nbool signbit(tv::tfloat32_t const &h)", target)
        self.assertIn("static TV_HOST_DEVICE_INLINE\nbool isnan(tv::tfloat32_t const &h)", target)
        self.assertIn("static TV_HOST_DEVICE_INLINE\nbool isfinite(tv::tfloat32_t const &h)", target)
        self.assertIn("static TV_HOST_DEVICE_INLINE\nbool isinf(tv::tfloat32_t const &h)", target)
        self.assertIn("static TV_HOST_DEVICE_INLINE\nbool isnormal(tv::tfloat32_t const &h)", target)
        self.assertIn("static TV_HOST_DEVICE_INLINE\nint fpclassify(tv::tfloat32_t const &h)", target)
        self.assertIn("static TV_HOST_DEVICE_INLINE\ntv::tfloat32_t sqrt(tv::tfloat32_t const &h)", target)
        self.assertIn("static TV_HOST_DEVICE_INLINE\ntv::tfloat32_t copysign(tv::tfloat32_t const &a, tv::tfloat32_t const &b)", target)

    def test_patch_cumm_dtype_headers_rewrites_previous_partial_patch_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            tf32_header = self._write_fake_cumm_tf32_partially_patched_header(expected_include)
            float8_header = self._write_fake_cumm_float8_partially_patched_header(expected_include)

            result = self.setup.patch_cumm_dtype_headers(venv_dir)
            tf32_patched = tf32_header.read_text(encoding="utf-8")
            float8_patched = float8_header.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "patched")
        self.assertEqual(tf32_patched, self._local_target_tf32_header_source())
        self.assertEqual(float8_patched, self._local_target_float8_header_source())

    def test_patch_cumm_dtype_headers_rewrites_previous_approximate_float8_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            self._write_fake_cumm_tf32_header(expected_include, source=self._canonical_tf32_header_source())
            float8_header = self._write_fake_cumm_float8_header(
                expected_include,
                source=self._approximate_guarded_float8_header_source(),
            )

            result = self.setup.patch_cumm_dtype_headers(venv_dir)
            float8_patched = float8_header.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "patched")
        self.assertEqual(float8_patched, self._local_target_float8_header_source())

    def test_patch_cumm_dtype_headers_rewrites_previous_repo_target_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            tf32_header = self._write_fake_cumm_tf32_header(
                expected_include,
                source=self._previous_local_target_tf32_header_source(),
            )
            float8_header = self._write_fake_cumm_float8_header(
                expected_include,
                source=self._legacy_local_target_float8_header_source(),
            )

            result = self.setup.patch_cumm_dtype_headers(venv_dir)
            tf32_patched = tf32_header.read_text(encoding="utf-8")
            float8_patched = float8_header.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "patched")
        self.assertEqual(tf32_patched, self._local_target_tf32_header_source())
        self.assertEqual(float8_patched, self._local_target_float8_header_source())

    def test_canonicalize_cumm_dtype_headers_returns_clear_statuses(self) -> None:
        patched_tf32, patched_tf32_status, patched_tf32_reason = self.setup._canonicalize_cumm_dtype_header(
            name="tf32",
            relative_path=self.setup.CUMM_TF32_HEADER_RELATIVE_PATH,
            legacy_fragments=self.setup.CUMM_TF32_HEADER_CANONICAL_FRAGMENTS,
            source=self._canonical_tf32_header_source(),
        )
        already_float8, already_float8_status, already_float8_reason = self.setup._canonicalize_cumm_dtype_header(
            name="float8",
            relative_path=self.setup.CUMM_FLOAT8_HEADER_RELATIVE_PATH,
            legacy_fragments=self.setup.CUMM_FLOAT8_HEADER_CANONICAL_FRAGMENTS,
            source=self._local_target_float8_header_source(),
        )
        failed_float8, failed_float8_status, failed_float8_reason = self.setup._canonicalize_cumm_dtype_header(
            name="float8",
            relative_path=self.setup.CUMM_FLOAT8_HEADER_RELATIVE_PATH,
            legacy_fragments=self.setup.CUMM_FLOAT8_HEADER_CANONICAL_FRAGMENTS,
            source="#pragma once\nnamespace std {\n}\n",
        )

        self.assertEqual(patched_tf32_status, "patched")
        self.assertIsNone(patched_tf32_reason)
        self.assertEqual(patched_tf32, self._local_target_tf32_header_source())
        self.assertEqual(already_float8_status, "already_patched")
        self.assertIsNone(already_float8_reason)
        self.assertEqual(already_float8, self._local_target_float8_header_source())
        self.assertEqual(failed_float8_status, "failed")
        self.assertEqual(failed_float8_reason, "unknown_header_source")
        self.assertEqual(failed_float8, "#pragma once\nnamespace std {\n}\n")

    def test_cumm_float8_local_target_contains_compiler_safe_forms(self) -> None:
        source = self._local_target_float8_header_source()
        constants_region = self._float8_static_constants_region(source)

        self.assertIn("static constexpr bool IS_E4M3 = (T == FloatEncoding::E4M3);", source)
        self.assertIn("static constexpr int FP8_NUM_EXPONENT_BITS = (T == FloatEncoding::E4M3) ? 4 : 5;", source)
        self.assertIn("static constexpr int FP8_NUM_MANTISSA_BITS = (T == FloatEncoding::E4M3) ? 3 : 2;", source)
        self.assertIn("static constexpr uint8_t  FP8_INFINITY_MASK = (T == FloatEncoding::E4M3) ? 0x78 : 0x7c;", source)
        self.assertIn("static constexpr int FP8_MAX_EXPONENT  = (T == FloatEncoding::E4M3) ?  7 :  15;", source)
        self.assertIn("static constexpr int FP8_MIN_EXPONENT  = (T == FloatEncoding::E4M3) ? -6 : -14;", source)
        self.assertIn("static constexpr int FP8_EXPONENT_BIAS = (T == FloatEncoding::E4M3) ?  7 :  15;", source)
        self.assertIn("static constexpr uint8_t  FP8_EXPONENT_MASK = uint8_t((1u << ((T == FloatEncoding::E4M3) ? 4 : 5)) - 1u);", source)
        self.assertIn("static constexpr uint8_t  FP8_MANTISSA_MASK = uint8_t((1u << ((T == FloatEncoding::E4M3) ? 3 : 2)) - 1u);", source)
        self.assertIn("static constexpr uint8_t FP8_MAX_FLT = ((T == FloatEncoding::E4M3) ? 0x7e : 0x7b);", source)
        self.assertNotIn("self_type::", constants_region)
        self.assertNotIn("static constexpr int FP8_NUM_EXPONENT_BITS = IS_E4M3 ? 4 : 5;", constants_region)
        self.assertNotIn("static constexpr int FP8_NUM_MANTISSA_BITS = IS_E4M3 ? 3 : 2;", constants_region)
        self.assertNotIn("static constexpr uint8_t  FP8_INFINITY_MASK = IS_E4M3 ? 0x78 : 0x7c;", constants_region)
        self.assertNotIn("static constexpr int FP8_MAX_EXPONENT  = IS_E4M3 ?  7 :  15;", constants_region)
        self.assertNotIn("static constexpr int FP8_MIN_EXPONENT  = IS_E4M3 ? -6 : -14;", constants_region)
        self.assertNotIn("static constexpr int FP8_EXPONENT_BIAS = IS_E4M3 ?  7 :  15;", constants_region)
        self.assertNotIn("static constexpr uint8_t  FP8_EXPONENT_MASK = (1 << self_type::FP8_NUM_EXPONENT_BITS) - 1;", constants_region)
        self.assertNotIn("static constexpr uint8_t  FP8_MANTISSA_MASK = (1 << self_type::FP8_NUM_MANTISSA_BITS) - 1;", constants_region)
        self.assertNotIn("static constexpr uint8_t FP8_MAX_FLT = (IS_E4M3 ? 0x7e : 0x7b);", constants_region)
        self.assertIn("using self_type = float8_base<T>;", source)
        self.assertIn("if (self_type::isnan(flt)) {", source)
        self.assertIn("if (self_type::isinf(flt)) {", source)
        self.assertIn("self_type::FP32_NUM_MANTISSA_BITS", source)
        self.assertIn("Base::FP8_NUM_MANTISSA_BITS", source)
        self.assertIn("#ifdef isfinite\n#undef isfinite\n#endif\n#ifdef isnan\n#undef isnan\n#endif\n#ifdef isinf\n#undef isinf\n#endif\n", source)
        self.assertNotIn("#define CUDA_NAMESPACE_STD", source)
        self.assertNotIn("namespace CUDA_NAMESPACE_STD {", source)
        self.assertNotIn("namespace cuda {\nnamespace std {", source)
        self.assertIn("#if !defined(__CUDACC_RTC__)\nnamespace std {", source)

    def test_cumm_tf32_local_target_declares_host_only_std_namespace_shape(self) -> None:
        source = self._local_target_tf32_header_source()

        self.assertNotIn("#define CUDA_NAMESPACE_STD", source)
        self.assertNotIn("namespace CUDA_NAMESPACE_STD {", source)
        self.assertIn("namespace tv {\n", source)
        self.assertIn("struct alignas(4) tfloat32_t {", source)
        self.assertIn("#if !defined(__CUDACC_RTC__)\nnamespace std {\n", source)
        self.assertNotIn("namespace cuda {\nnamespace std {", source)
        self.assertIn("template <> struct numeric_limits<tv::tfloat32_t> {", source)
        self.assertIn("} // namespace std\n#endif\n", source)
        self.assertIn("#ifdef signbit\n#undef signbit\n#endif\n#ifdef isnan\n#undef isnan\n#endif\n#ifdef isfinite\n#undef isfinite\n#endif\n#ifdef isinf\n#undef isinf\n#endif\n#ifdef isnormal\n#undef isnormal\n#endif\n#ifdef fpclassify\n#undef fpclassify\n#endif\n#ifdef sqrt\n#undef sqrt\n#endif\n#ifdef copysign\n#undef copysign\n#endif\n", source)
        self.assertNotIn("#if !defined(__CUDACC_RTC__)\n#ifdef signbit", source)

    def test_cumm_tf32_local_target_differs_from_previous_broken_repo_target(self) -> None:
        source = self._local_target_tf32_header_source()
        latest_live_broken_source = self._latest_live_broken_tf32_header_source()
        previous_source = self._previous_local_target_tf32_header_source()

        self.assertNotEqual(source, latest_live_broken_source)
        self.assertNotEqual(source, previous_source)
        self.assertNotIn("#undef signbit", latest_live_broken_source)
        self.assertIn("namespace std {\n\n#if !defined(__CUDACC_RTC__)", previous_source)
        self.assertIn("#if !defined(__CUDACC_RTC__)\nnamespace std {\n", source)

    def test_cumm_dtype_local_targets_avoid_latest_live_error_constructs(self) -> None:
        tf32_source = self._local_target_tf32_header_source()
        float8_source = self._local_target_float8_header_source()
        float8_constants = self._float8_static_constants_region(float8_source)

        self.assertNotIn("namespace CUDA_NAMESPACE_STD {", tf32_source)
        self.assertNotIn("namespace cuda {\nnamespace std {", tf32_source)
        self.assertNotIn("#define CUDA_NAMESPACE_STD", tf32_source)
        self.assertNotIn("namespace CUDA_NAMESPACE_STD {", float8_source)
        self.assertNotIn("namespace cuda {\nnamespace std {", float8_source)
        self.assertNotIn("#define CUDA_NAMESPACE_STD", float8_source)
        self.assertIn("static constexpr int FP8_NUM_EXPONENT_BITS = (T == FloatEncoding::E4M3) ? 4 : 5;", float8_source)
        self.assertIn("static constexpr int FP8_NUM_MANTISSA_BITS = (T == FloatEncoding::E4M3) ? 3 : 2;", float8_source)
        self.assertIn("static constexpr uint8_t  FP8_INFINITY_MASK = (T == FloatEncoding::E4M3) ? 0x78 : 0x7c;", float8_source)
        self.assertIn("static constexpr int FP8_MAX_EXPONENT  = (T == FloatEncoding::E4M3) ?  7 :  15;", float8_source)
        self.assertIn("static constexpr int FP8_MIN_EXPONENT  = (T == FloatEncoding::E4M3) ? -6 : -14;", float8_source)
        self.assertIn("static constexpr int FP8_EXPONENT_BIAS = (T == FloatEncoding::E4M3) ?  7 :  15;", float8_source)
        self.assertIn("static constexpr uint8_t FP8_MAX_FLT = ((T == FloatEncoding::E4M3) ? 0x7e : 0x7b);", float8_source)
        self.assertNotIn("self_type::", float8_constants)
        self.assertNotIn("int8_t exp = uint8_t(((s >> FP32_NUM_MANTISSA_BITS) & 0xff) - FP32_EXPONENT_BIAS);", float8_source)
        self.assertNotIn("return int((storage >> FP8_NUM_MANTISSA_BITS) & Base::FP8_EXPONENT_MASK);", float8_source)
        self.assertIn("#undef isfinite", float8_source)
        self.assertIn("#undef isnan", float8_source)
        self.assertIn("#undef isinf", float8_source)

    def test_canonicalize_cumm_dtype_headers_repairs_latest_live_broken_targets(self) -> None:
        tf32_patched, tf32_status, tf32_reason = self.setup._canonicalize_cumm_dtype_header(
            name="tf32",
            relative_path=self.setup.CUMM_TF32_HEADER_RELATIVE_PATH,
            legacy_fragments=self.setup.CUMM_TF32_HEADER_CANONICAL_FRAGMENTS,
            source=self._latest_live_broken_tf32_header_source(),
        )
        float8_patched, float8_status, float8_reason = self.setup._canonicalize_cumm_dtype_header(
            name="float8",
            relative_path=self.setup.CUMM_FLOAT8_HEADER_RELATIVE_PATH,
            legacy_fragments=self.setup.CUMM_FLOAT8_HEADER_CANONICAL_FRAGMENTS,
            source=self._latest_live_broken_float8_header_source(),
        )

        self.assertEqual(tf32_status, "patched")
        self.assertIsNone(tf32_reason)
        self.assertEqual(tf32_patched, self._local_target_tf32_header_source())
        self.assertNotEqual(tf32_patched, self._latest_live_broken_tf32_header_source())
        self.assertNotEqual(tf32_patched, self._previous_local_target_tf32_header_source())
        self.assertEqual(float8_status, "patched")
        self.assertIsNone(float8_reason)
        self.assertEqual(float8_patched, self._local_target_float8_header_source())
        self.assertNotEqual(float8_patched, self._latest_live_broken_float8_header_source())

    def test_canonicalize_cumm_float8_header_repairs_latest_live_self_type_constants_target(self) -> None:
        patched, status, reason = self.setup._canonicalize_cumm_dtype_header(
            name="float8",
            relative_path=self.setup.CUMM_FLOAT8_HEADER_RELATIVE_PATH,
            legacy_fragments=self.setup.CUMM_FLOAT8_HEADER_CANONICAL_FRAGMENTS,
            source=self._latest_live_broken_float8_header_source(),
        )

        self.assertEqual(status, "patched")
        self.assertIsNone(reason)
        self.assertEqual(patched, self._local_target_float8_header_source())
        self.assertIn("self_type::IS_E4M3", self._latest_live_broken_float8_header_source())
        self.assertNotIn("self_type::", self._float8_static_constants_region(patched))

    def test_canonicalize_cumm_tf32_header_accepts_previous_target_idempotently(self) -> None:
        patched, status, reason = self.setup._canonicalize_cumm_dtype_header(
            name="tf32",
            relative_path=self.setup.CUMM_TF32_HEADER_RELATIVE_PATH,
            legacy_fragments=self.setup.CUMM_TF32_HEADER_CANONICAL_FRAGMENTS,
            source=self._previous_local_target_tf32_header_source(),
        )

        self.assertEqual(status, "patched")
        self.assertIsNone(reason)
        self.assertEqual(patched, self._local_target_tf32_header_source())

        again, again_status, again_reason = self.setup._canonicalize_cumm_dtype_header(
            name="tf32",
            relative_path=self.setup.CUMM_TF32_HEADER_RELATIVE_PATH,
            legacy_fragments=self.setup.CUMM_TF32_HEADER_CANONICAL_FRAGMENTS,
            source=patched,
        )

        self.assertEqual(again_status, "already_patched")
        self.assertIsNone(again_reason)
        self.assertEqual(again, patched)

    def test_canonicalize_cumm_float8_header_accepts_current_broken_repo_target_idempotently(self) -> None:
        patched, status, reason = self.setup._canonicalize_cumm_dtype_header(
            name="float8",
            relative_path=self.setup.CUMM_FLOAT8_HEADER_RELATIVE_PATH,
            legacy_fragments=self.setup.CUMM_FLOAT8_HEADER_CANONICAL_FRAGMENTS,
            source=self._previous_local_target_float8_header_source(),
        )

        self.assertEqual(status, "patched")
        self.assertIsNone(reason)
        self.assertEqual(patched, self._local_target_float8_header_source())

        again, again_status, again_reason = self.setup._canonicalize_cumm_dtype_header(
            name="float8",
            relative_path=self.setup.CUMM_FLOAT8_HEADER_RELATIVE_PATH,
            legacy_fragments=self.setup.CUMM_FLOAT8_HEADER_CANONICAL_FRAGMENTS,
            source=patched,
        )

        self.assertEqual(again_status, "already_patched")
        self.assertIsNone(again_reason)
        self.assertEqual(again, patched)

    def test_patch_cumm_dtype_headers_fails_closed_on_unknown_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir, _site_packages, _package_root, expected_include, sentinel = self._create_fake_cumm_layout(Path(temp_dir))
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("// tensorview\n", encoding="utf-8")
            self._write_fake_cumm_tf32_header(expected_include)
            float8_header = self._write_fake_cumm_float8_header(expected_include, source="#pragma once\nnamespace std {\n}\n")
            before = float8_header.read_text(encoding="utf-8")

            result = self.setup.patch_cumm_dtype_headers(venv_dir)
            after = float8_header.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "unknown_float8_header_source")
        self.assertEqual(before, after)

    def test_run_native_runtime_phase_auto_does_not_attempt_install(self) -> None:
        with mock.patch.object(
            self.setup,
            "probe_native_runtime",
            return_value=self.setup.NativeRuntimeProbe(
                ready=False,
                status="blocked",
                managed_python="/tmp/python",
                imports={},
                missing_modules=("spconv",),
                blocked_imports=("spconv.pytorch",),
                message="blocked",
                next_action="repair",
            ),
        ) as probe_mock:
            installer = mock.Mock()
            result = self.setup.run_native_runtime_phase(
                venv_dir=Path("/tmp/venv"),
                managed_python=Path("/tmp/python"),
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                native_mode="auto",
                installer=installer,
            )

        self.assertEqual(probe_mock.call_count, 1)
        installer.assert_not_called()
        self.assertFalse(result["attempted_install"])
        self.assertEqual(result["mode"], "auto")
        self.assertEqual(result["install_strategy"], "linux-arm64-source-build")

    def test_run_native_runtime_phase_install_can_use_monkeypatched_runner(self) -> None:
        plan = self.setup.NativeRuntimePlan(
            native_mode="install",
            host=self.setup.NativeHostLane(
                system="Linux",
                machine="arm64",
                platform_key="linux/arm64",
                python_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                lane="linux-arm64-cu128",
                state="supported",
                reason="ok",
            ),
            desired_cuda_label="cu128",
            toolkit_root="/usr/local/cuda-12.8",
            toolkit_include_dir="/usr/local/cuda-12.8/include",
            toolkit_nvcc="/usr/local/cuda-12.8/bin/nvcc",
            nvcc_version="12.8",
            toolkit_status="matched",
            toolkit_reason="ok",
            required_imports=("spconv.pytorch", "torch_scatter", "torch_cluster"),
            related_packages=("cumm",),
            install_steps=(
                self.setup.NativeInstallStep(
                    package="native-build-prerequisites",
                    requirement="pccm>=0.4.16",
                    strategy="pip-install-prerequisites",
                    status="planned",
                    pip_args=("install", "pccm>=0.4.16"),
                    env={"TORCH_CUDA_ARCH_LIST": "10.0", "CUMM_CUDA_ARCH_LIST": "10.0"},
                ),
                self.setup.NativeInstallStep(
                    package="torch_scatter",
                    requirement="torch_scatter",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "torch_scatter==test"),
                    env={"TORCH_CUDA_ARCH_LIST": "10.0", "CUMM_CUDA_ARCH_LIST": "10.0"},
                ),
                self.setup.NativeInstallStep(
                    package="torch_cluster",
                    requirement="torch_cluster",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "torch-cluster==test"),
                    env={"TORCH_CUDA_ARCH_LIST": "10.0", "CUMM_CUDA_ARCH_LIST": "10.0"},
                ),
            ),
            install_strategy="linux-arm64-source-build",
            automatic_install_supported=True,
            status="planned",
            message="install",
            next_action="install",
        )
        blocked_probe = self.setup.NativeRuntimeProbe(
            ready=False,
            status="blocked",
            managed_python="/tmp/python",
            imports={},
            missing_modules=("torch_scatter", "torch_cluster"),
            blocked_imports=("torch_scatter", "torch_cluster"),
            message="blocked",
            next_action="repair",
        )
        ready_probe = self.setup.NativeRuntimeProbe(
            ready=True,
            status="ready",
            managed_python="/tmp/python",
            imports={
                "torch_scatter": {"ready": True, "status": "ready"},
                "torch_cluster": {"ready": True, "status": "ready"},
            },
            missing_modules=(),
            blocked_imports=(),
            message="ready",
            next_action="none",
        )

        with (
            mock.patch.object(self.setup, "build_native_runtime_plan", return_value=plan),
            mock.patch.object(self.setup, "probe_native_runtime", side_effect=[blocked_probe, ready_probe]) as probe_mock,
        ):
            installer = mock.Mock(side_effect=[
                {"package": "native-build-prerequisites", "status": "installed"},
                {"package": "torch_scatter", "status": "installed"},
                {"package": "torch_cluster", "status": "installed"},
            ])
            result = self.setup.run_native_runtime_phase(
                venv_dir=Path("/tmp/venv"),
                managed_python=Path("/tmp/python"),
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                native_mode="install",
                installer=installer,
            )

        self.assertEqual(probe_mock.call_count, 2)
        self.assertEqual(installer.call_count, 3)
        self.assertEqual([call.kwargs["step"].package for call in installer.call_args_list], [
            "native-build-prerequisites",
            "torch_scatter",
            "torch_cluster",
        ])
        self.assertTrue(result["attempted_install"])
        self.assertTrue(result["ready"])

    def test_run_native_runtime_phase_install_failure_returns_structured_degraded_result(self) -> None:
        plan = self.setup.NativeRuntimePlan(
            native_mode="install",
            host=self.setup.NativeHostLane(
                system="Linux",
                machine="arm64",
                platform_key="linux/arm64",
                python_tag="cp311",
                gpu_sm=89,
                cuda_version=124,
                lane="linux-arm64-cu124",
                state="supported",
                reason="ok",
            ),
            desired_cuda_label="cu124",
            toolkit_root="/usr/local/cuda-12.4",
            nvcc_version="12.4",
            toolkit_status="matched",
            toolkit_reason="ok",
            required_imports=("spconv.pytorch", "torch_scatter", "torch_cluster"),
            related_packages=("cumm",),
            install_steps=(
                self.setup.NativeInstallStep(
                    package="torch_scatter",
                    requirement="torch-scatter==2.1.2",
                    strategy="pip-source-build",
                    status="planned",
                    pip_args=("install", "torch-scatter==2.1.2"),
                    env={"TORCH_CUDA_ARCH_LIST": "8.9", "CUMM_CUDA_ARCH_LIST": "8.9"},
                ),
            ),
            install_strategy="linux-arm64-source-build",
            automatic_install_supported=True,
            status="planned",
            message="install",
            next_action="install",
        )
        blocked_probe = self.setup.NativeRuntimeProbe(
            ready=False,
            status="blocked",
            managed_python="/tmp/python",
            imports={},
            missing_modules=("torch_scatter",),
            blocked_imports=("torch_scatter",),
            message="blocked",
            next_action="repair",
        )

        with (
            mock.patch.object(self.setup, "build_native_runtime_plan", return_value=plan),
            mock.patch.object(self.setup, "probe_native_runtime", return_value=blocked_probe) as probe_mock,
        ):
            installer = mock.Mock(
                return_value={
                    "package": "torch_scatter",
                    "status": "failed",
                    "returncode": 1,
                    "stdout_tail": ["build log"],
                    "stderr_tail": ["compile error"],
                }
            )
            result = self.setup.run_native_runtime_phase(
                venv_dir=Path("/tmp/venv"),
                managed_python=Path("/tmp/python"),
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=89,
                cuda_version=124,
                native_mode="install",
                installer=installer,
            )

        self.assertEqual(probe_mock.call_count, 1)
        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["ready"])
        self.assertTrue(result["attempted_install"])
        self.assertEqual(result["install_results"][0]["stderr_tail"], ["compile error"])

    def test_run_native_runtime_phase_install_stays_blocked_when_toolkit_missing(self) -> None:
        plan = self.setup.NativeRuntimePlan(
            native_mode="install",
            host=self.setup.NativeHostLane(
                system="Linux",
                machine="arm64",
                platform_key="linux/arm64",
                python_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                lane="linux-arm64-cu128",
                state="supported",
                reason="ok",
            ),
            desired_cuda_label="cu128",
            toolkit_root=None,
            nvcc_version=None,
            toolkit_status="blocked",
            toolkit_reason="CUDA 12.8 toolkit missing",
            required_imports=("spconv.pytorch", "torch_scatter", "torch_cluster"),
            related_packages=("cumm",),
            install_steps=(
                self.setup.NativeInstallStep(
                    package="torch_scatter",
                    requirement="torch-scatter==2.1.2",
                    strategy="pip-source-build",
                    status="blocked",
                    reason="CUDA 12.8 toolkit missing",
                ),
            ),
            install_strategy="linux-arm64-source-build",
            automatic_install_supported=False,
            status="blocked",
            message="blocked",
            next_action="Install CUDA 12.8 toolkit and retry.",
        )
        blocked_probe = self.setup.NativeRuntimeProbe(
            ready=False,
            status="blocked",
            managed_python="/tmp/python",
            imports={},
            missing_modules=("torch_scatter",),
            blocked_imports=("torch_scatter",),
            message="blocked",
            next_action="repair",
        )

        with (
            mock.patch.object(self.setup, "build_native_runtime_plan", return_value=plan),
            mock.patch.object(self.setup, "probe_native_runtime", return_value=blocked_probe),
        ):
            installer = mock.Mock()
            result = self.setup.run_native_runtime_phase(
                venv_dir=Path("/tmp/venv"),
                managed_python=Path("/tmp/python"),
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                native_mode="install",
                installer=installer,
            )

        installer.assert_not_called()
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["attempted_install"])
        self.assertEqual(result["next_action"], "Install CUDA 12.8 toolkit and retry.")

    def test_run_native_runtime_phase_repairs_cumm_headers_before_spconv_install(self) -> None:
        plan = self.setup.NativeRuntimePlan(
            native_mode="install",
            host=self.setup.NativeHostLane(
                system="Linux",
                machine="arm64",
                platform_key="linux/arm64",
                python_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                lane="linux-arm64-cu128",
                state="supported",
                reason="ok",
            ),
            desired_cuda_label="cu128",
            toolkit_root="/usr/local/cuda-12.8",
            toolkit_include_dir="/usr/local/cuda-12.8/include",
            toolkit_nvcc="/usr/local/cuda-12.8/bin/nvcc",
            nvcc_version="12.8",
            toolkit_status="matched",
            toolkit_reason="ok",
            required_imports=("spconv.pytorch", "torch_scatter", "torch_cluster"),
            related_packages=("cumm",),
            install_steps=(
                self.setup.NativeInstallStep(
                    package="cumm",
                    requirement="cumm",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "cumm"),
                ),
                self.setup.NativeInstallStep(
                    package="spconv",
                    requirement="spconv",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "spconv"),
                ),
            ),
            install_strategy="linux-arm64-source-build",
            automatic_install_supported=True,
            status="planned",
            message="install",
            next_action="install",
        )
        blocked_probe = self.setup.NativeRuntimeProbe(
            ready=False,
            status="blocked",
            managed_python="/tmp/python",
            imports={},
            missing_modules=("spconv", "cumm"),
            blocked_imports=("spconv.pytorch",),
            message="blocked",
            next_action="repair",
        )
        ready_probe = self.setup.NativeRuntimeProbe(
            ready=True,
            status="ready",
            managed_python="/tmp/python",
            imports={"spconv.pytorch": {"ready": True, "status": "ready"}},
            missing_modules=(),
            blocked_imports=(),
            message="ready",
            next_action="none",
        )
        events: list[str] = []

        def installer(*, venv_dir: Path, step: object) -> dict[str, str]:
            events.append(f"install:{step.package}")
            return {"package": step.package, "status": "installed"}

        def repairer(*, venv_dir: Path, extension_dir: Path) -> dict[str, str]:
            events.append("repair:cumm")
            return {"package": "cumm-tensorview-headers", "status": "repaired", "next_action": "none"}

        def validator(*, managed_python: Path) -> dict[str, object]:
            events.append("validate:cumm.core_cc")
            return {
                "package": "cumm-core-cc",
                "module": "cumm.core_cc",
                "ready": True,
                "status": "ready",
                "next_action": "none",
            }

        def patcher(*, venv_dir: Path) -> dict[str, object]:
            events.append("patch:cumm-subint")
            return {
                "package": "cumm-subint-header-patch",
                "status": "patched",
                "next_action": "none",
            }

        def half_patcher(*, venv_dir: Path) -> dict[str, object]:
            events.append("patch:cumm-half")
            return {
                "package": "cumm-half-header-patch",
                "status": "patched",
                "next_action": "none",
            }

        def bfloat16_patcher(*, venv_dir: Path) -> dict[str, object]:
            events.append("patch:cumm-bfloat16")
            return {
                "package": "cumm-bfloat16-header-patch",
                "status": "patched",
                "next_action": "none",
            }

        def dtype_patcher(*, venv_dir: Path) -> dict[str, object]:
            events.append("patch:cumm-dtypes")
            return {
                "package": "cumm-dtype-header-patch",
                "status": "patched",
                "next_action": "none",
            }

        def cuda_resolver_patcher(*, venv_dir: Path, toolkit_root: str | None, include_dir: str | None, nvcc_path: str | None) -> dict[str, object]:
            events.append("patch:cumm-cuda-resolver")
            self.assertEqual(toolkit_root, "/usr/local/cuda-12.8")
            self.assertEqual(include_dir, "/usr/local/cuda-12.8/include")
            self.assertEqual(nvcc_path, "/usr/local/cuda-12.8/bin/nvcc")
            return {
                "package": "cumm-cuda-resolver-patch",
                "status": "patched",
                "next_action": "none",
                "selected_toolkit_root": toolkit_root,
                "selected_include_dir": include_dir,
                "selected_nvcc_path": nvcc_path,
                "no_global_cuda_mutation": True,
            }

        def spconv_validator(*, managed_python: Path) -> dict[str, object]:
            events.append("validate:spconv.core_cc")
            return {
                "package": "spconv-core-cc",
                "module": "spconv.core_cc",
                "ready": True,
                "status": "ready",
                "next_action": "none",
            }

        with (
            mock.patch.object(self.setup, "build_native_runtime_plan", return_value=plan),
            mock.patch.object(self.setup, "probe_native_runtime", side_effect=[blocked_probe, ready_probe]) as probe_mock,
        ):
            result = self.setup.run_native_runtime_phase(
                venv_dir=Path("/tmp/venv"),
                managed_python=Path("/tmp/python"),
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                native_mode="install",
                installer=installer,
                cumm_header_repairer=repairer,
                cumm_core_validator=validator,
                cumm_subint_header_patcher=patcher,
                cumm_half_header_patcher=half_patcher,
                cumm_bfloat16_header_patcher=bfloat16_patcher,
                cumm_dtype_header_patcher=dtype_patcher,
                cumm_cuda_resolver_patcher=cuda_resolver_patcher,
                spconv_core_validator=spconv_validator,
            )

        self.assertEqual(probe_mock.call_count, 2)
        self.assertEqual(events, ["install:cumm", "repair:cumm", "validate:cumm.core_cc", "patch:cumm-subint", "patch:cumm-half", "patch:cumm-bfloat16", "patch:cumm-dtypes", "patch:cumm-cuda-resolver", "install:spconv", "validate:spconv.core_cc"])
        self.assertEqual([entry["package"] for entry in result["install_results"]], [
            "cumm",
            "cumm-tensorview-headers",
            "cumm-core-cc",
            "cumm-subint-header-patch",
            "cumm-half-header-patch",
            "cumm-bfloat16-header-patch",
            "cumm-dtype-header-patch",
            "cumm-cuda-resolver-patch",
            "spconv",
            "spconv-core-cc",
        ])
        self.assertTrue(result["ready"])

    def test_run_native_runtime_phase_stops_before_spconv_when_cumm_cuda_resolver_patch_fails(self) -> None:
        plan = self.setup.NativeRuntimePlan(
            native_mode="install",
            host=self.setup.NativeHostLane(
                system="Linux",
                machine="arm64",
                platform_key="linux/arm64",
                python_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                lane="linux-arm64-cu128",
                state="supported",
                reason="ok",
            ),
            desired_cuda_label="cu128",
            toolkit_root="/usr/local/cuda-12.8",
            toolkit_include_dir="/usr/local/cuda-12.8/include",
            toolkit_nvcc="/usr/local/cuda-12.8/bin/nvcc",
            nvcc_version="12.8",
            toolkit_status="matched",
            toolkit_reason="ok",
            required_imports=("spconv.pytorch", "torch_scatter", "torch_cluster"),
            related_packages=("cumm",),
            install_steps=(
                self.setup.NativeInstallStep(
                    package="cumm",
                    requirement="cumm",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "cumm"),
                ),
                self.setup.NativeInstallStep(
                    package="spconv",
                    requirement="spconv",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "spconv"),
                ),
            ),
            install_strategy="linux-arm64-source-build",
            automatic_install_supported=True,
            status="planned",
            message="install",
            next_action="install",
        )
        blocked_probe = self.setup.NativeRuntimeProbe(
            ready=False,
            status="blocked",
            managed_python="/tmp/python",
            imports={},
            missing_modules=("spconv", "cumm"),
            blocked_imports=("spconv.pytorch",),
            message="blocked",
            next_action="repair",
        )
        events: list[str] = []

        def installer(*, venv_dir: Path, step: object) -> dict[str, str]:
            events.append(f"install:{step.package}")
            return {"package": step.package, "status": "installed"}

        def repairer(*, venv_dir: Path, extension_dir: Path) -> dict[str, str]:
            events.append("repair:cumm")
            return {"package": "cumm-tensorview-headers", "status": "repaired", "next_action": "none"}

        def validator(*, managed_python: Path) -> dict[str, object]:
            events.append("validate:cumm.core_cc")
            return {
                "package": "cumm-core-cc",
                "module": "cumm.core_cc",
                "ready": True,
                "status": "ready",
                "next_action": "none",
            }

        with (
            mock.patch.object(self.setup, "build_native_runtime_plan", return_value=plan),
            mock.patch.object(self.setup, "probe_native_runtime", return_value=blocked_probe) as probe_mock,
        ):
            result = self.setup.run_native_runtime_phase(
                venv_dir=Path("/tmp/venv"),
                managed_python=Path("/tmp/python"),
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                native_mode="install",
                installer=installer,
                cumm_header_repairer=repairer,
                cumm_core_validator=validator,
                cumm_subint_header_patcher=lambda *, venv_dir: {
                    "package": "cumm-subint-header-patch",
                    "status": "patched",
                    "next_action": "none",
                },
                cumm_half_header_patcher=lambda *, venv_dir: {
                    "package": "cumm-half-header-patch",
                    "status": "patched",
                    "next_action": "none",
                },
                cumm_bfloat16_header_patcher=lambda *, venv_dir: {
                    "package": "cumm-bfloat16-header-patch",
                    "status": "patched",
                    "next_action": "none",
                },
                cumm_dtype_header_patcher=lambda *, venv_dir: {
                    "package": "cumm-dtype-header-patch",
                    "status": "patched",
                    "next_action": "none",
                },
                cumm_cuda_resolver_patcher=lambda **_: {
                    "package": "cumm-cuda-resolver-patch",
                    "status": "failed",
                    "reason": "unknown_cumm_cuda_resolver_source",
                    "next_action": "inspect cumm/common.py",
                },
            )

        self.assertEqual(probe_mock.call_count, 1)
        self.assertEqual(events, ["install:cumm", "repair:cumm", "validate:cumm.core_cc"])
        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["ready"])
        self.assertIn("CUDA resolver patch failed", result["message"])
        self.assertEqual(result["next_action"], "inspect cumm/common.py")
        self.assertEqual([entry["package"] for entry in result["install_results"]], [
            "cumm",
            "cumm-tensorview-headers",
            "cumm-core-cc",
            "cumm-subint-header-patch",
            "cumm-half-header-patch",
            "cumm-bfloat16-header-patch",
            "cumm-dtype-header-patch",
            "cumm-cuda-resolver-patch",
        ])

    def test_run_native_runtime_phase_stops_before_spconv_when_cumm_core_cc_validation_fails(self) -> None:
        plan = self.setup.NativeRuntimePlan(
            native_mode="install",
            host=self.setup.NativeHostLane(
                system="Linux",
                machine="arm64",
                platform_key="linux/arm64",
                python_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                lane="linux-arm64-cu128",
                state="supported",
                reason="ok",
            ),
            desired_cuda_label="cu128",
            toolkit_root="/usr/local/cuda-12.8",
            nvcc_version="12.8",
            toolkit_status="matched",
            toolkit_reason="ok",
            required_imports=("spconv.pytorch", "torch_scatter", "torch_cluster"),
            related_packages=("cumm",),
            install_steps=(
                self.setup.NativeInstallStep(
                    package="cumm",
                    requirement="cumm",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "cumm"),
                ),
                self.setup.NativeInstallStep(
                    package="spconv",
                    requirement="spconv",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "spconv"),
                ),
            ),
            install_strategy="linux-arm64-source-build",
            automatic_install_supported=True,
            status="planned",
            message="install",
            next_action="install",
        )
        blocked_probe = self.setup.NativeRuntimeProbe(
            ready=False,
            status="blocked",
            managed_python="/tmp/python",
            imports={},
            missing_modules=("spconv", "cumm"),
            blocked_imports=("spconv.pytorch",),
            message="blocked",
            next_action="repair",
        )
        events: list[str] = []

        def installer(*, venv_dir: Path, step: object) -> dict[str, str]:
            events.append(f"install:{step.package}")
            return {"package": step.package, "status": "installed"}

        def repairer(*, venv_dir: Path, extension_dir: Path) -> dict[str, str]:
            events.append("repair:cumm")
            return {"package": "cumm-tensorview-headers", "status": "repaired", "next_action": "none"}

        def validator(*, managed_python: Path) -> dict[str, object]:
            events.append("validate:cumm.core_cc")
            return {
                "package": "cumm-core-cc",
                "module": "cumm.core_cc",
                "ready": False,
                "status": "missing",
                "error": "No module named 'cumm.core_cc'",
                "traceback_tail": ["ModuleNotFoundError: No module named 'cumm.core_cc'"],
                "next_action": "Repair the cumm install so cumm.core_cc imports cleanly before retrying spconv.",
            }

        with (
            mock.patch.object(self.setup, "build_native_runtime_plan", return_value=plan),
            mock.patch.object(self.setup, "probe_native_runtime", return_value=blocked_probe) as probe_mock,
        ):
            result = self.setup.run_native_runtime_phase(
                venv_dir=Path("/tmp/venv"),
                managed_python=Path("/tmp/python"),
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                native_mode="install",
                installer=installer,
                cumm_header_repairer=repairer,
                cumm_core_validator=validator,
            )

        self.assertEqual(probe_mock.call_count, 1)
        self.assertEqual(events, ["install:cumm", "repair:cumm", "validate:cumm.core_cc"])
        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["ready"])
        self.assertIn("cumm.core_cc", result["message"])
        self.assertEqual([entry["package"] for entry in result["install_results"]], [
            "cumm",
            "cumm-tensorview-headers",
            "cumm-core-cc",
        ])

    def test_run_native_runtime_phase_stops_before_spconv_when_cumm_subint_patch_fails(self) -> None:
        plan = self.setup.NativeRuntimePlan(
            native_mode="install",
            host=self.setup.NativeHostLane(
                system="Linux",
                machine="arm64",
                platform_key="linux/arm64",
                python_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                lane="linux-arm64-cu128",
                state="supported",
                reason="ok",
            ),
            desired_cuda_label="cu128",
            toolkit_root="/usr/local/cuda-12.8",
            nvcc_version="12.8",
            toolkit_status="matched",
            toolkit_reason="ok",
            required_imports=("spconv.pytorch", "torch_scatter", "torch_cluster"),
            related_packages=("cumm",),
            install_steps=(
                self.setup.NativeInstallStep(
                    package="cumm",
                    requirement="cumm",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "cumm"),
                ),
                self.setup.NativeInstallStep(
                    package="spconv",
                    requirement="spconv",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "spconv"),
                ),
            ),
            install_strategy="linux-arm64-source-build",
            automatic_install_supported=True,
            status="planned",
            message="install",
            next_action="install",
        )
        blocked_probe = self.setup.NativeRuntimeProbe(
            ready=False,
            status="blocked",
            managed_python="/tmp/python",
            imports={},
            missing_modules=("spconv", "cumm"),
            blocked_imports=("spconv.pytorch",),
            message="blocked",
            next_action="repair",
        )
        events: list[str] = []

        def installer(*, venv_dir: Path, step: object) -> dict[str, str]:
            events.append(f"install:{step.package}")
            return {"package": step.package, "status": "installed"}

        def repairer(*, venv_dir: Path, extension_dir: Path) -> dict[str, str]:
            events.append("repair:cumm")
            return {"package": "cumm-tensorview-headers", "status": "repaired", "next_action": "none"}

        def validator(*, managed_python: Path) -> dict[str, object]:
            events.append("validate:cumm.core_cc")
            return {
                "package": "cumm-core-cc",
                "module": "cumm.core_cc",
                "ready": True,
                "status": "ready",
                "next_action": "none",
            }

        def patcher(*, venv_dir: Path) -> dict[str, object]:
            events.append("patch:cumm-subint")
            return {
                "package": "cumm-subint-header-patch",
                "status": "failed",
                "reason": "unknown_subint_header_source",
                "next_action": "inspect subint header",
            }

        with (
            mock.patch.object(self.setup, "build_native_runtime_plan", return_value=plan),
            mock.patch.object(self.setup, "probe_native_runtime", return_value=blocked_probe) as probe_mock,
        ):
            result = self.setup.run_native_runtime_phase(
                venv_dir=Path("/tmp/venv"),
                managed_python=Path("/tmp/python"),
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                native_mode="install",
                installer=installer,
                cumm_header_repairer=repairer,
                cumm_core_validator=validator,
                cumm_subint_header_patcher=patcher,
            )

        self.assertEqual(probe_mock.call_count, 1)
        self.assertEqual(events, ["install:cumm", "repair:cumm", "validate:cumm.core_cc", "patch:cumm-subint"])
        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["ready"])
        self.assertIn("subint header patch failed", result["message"])
        self.assertEqual(result["next_action"], "inspect subint header")
        self.assertEqual([entry["package"] for entry in result["install_results"]], [
            "cumm",
            "cumm-tensorview-headers",
            "cumm-core-cc",
            "cumm-subint-header-patch",
        ])

    def test_run_native_runtime_phase_stops_before_spconv_when_cumm_dtype_patch_fails(self) -> None:
        plan = self.setup.NativeRuntimePlan(
            native_mode="install",
            host=self.setup.NativeHostLane(
                system="Linux",
                machine="arm64",
                platform_key="linux/arm64",
                python_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                lane="linux-arm64-cu128",
                state="supported",
                reason="ok",
            ),
            desired_cuda_label="cu128",
            toolkit_root="/usr/local/cuda-12.8",
            nvcc_version="12.8",
            toolkit_status="matched",
            toolkit_reason="ok",
            required_imports=("spconv.pytorch", "torch_scatter", "torch_cluster"),
            related_packages=("cumm",),
            install_steps=(
                self.setup.NativeInstallStep(
                    package="cumm",
                    requirement="cumm",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "cumm"),
                ),
                self.setup.NativeInstallStep(
                    package="spconv",
                    requirement="spconv",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "spconv"),
                ),
            ),
            install_strategy="linux-arm64-source-build",
            automatic_install_supported=True,
            status="planned",
            message="install",
            next_action="install",
        )
        blocked_probe = self.setup.NativeRuntimeProbe(
            ready=False,
            status="blocked",
            managed_python="/tmp/python",
            imports={},
            missing_modules=("spconv", "cumm"),
            blocked_imports=("spconv.pytorch",),
            message="blocked",
            next_action="repair",
        )
        events: list[str] = []

        def installer(*, venv_dir: Path, step: object) -> dict[str, str]:
            events.append(f"install:{step.package}")
            return {"package": step.package, "status": "installed"}

        def repairer(*, venv_dir: Path, extension_dir: Path) -> dict[str, str]:
            events.append("repair:cumm")
            return {"package": "cumm-tensorview-headers", "status": "repaired", "next_action": "none"}

        def validator(*, managed_python: Path) -> dict[str, object]:
            events.append("validate:cumm.core_cc")
            return {
                "package": "cumm-core-cc",
                "module": "cumm.core_cc",
                "ready": True,
                "status": "ready",
                "next_action": "none",
            }

        def subint_patcher(*, venv_dir: Path) -> dict[str, object]:
            events.append("patch:cumm-subint")
            return {
                "package": "cumm-subint-header-patch",
                "status": "patched",
                "next_action": "none",
            }

        def dtype_patcher(*, venv_dir: Path) -> dict[str, object]:
            events.append("patch:cumm-dtypes")
            return {
                "package": "cumm-dtype-header-patch",
                "status": "failed",
                "reason": "unknown_float8_header_source",
                "next_action": "inspect float8 header",
            }

        with (
            mock.patch.object(self.setup, "build_native_runtime_plan", return_value=plan),
            mock.patch.object(self.setup, "probe_native_runtime", return_value=blocked_probe) as probe_mock,
        ):
            result = self.setup.run_native_runtime_phase(
                venv_dir=Path("/tmp/venv"),
                managed_python=Path("/tmp/python"),
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                native_mode="install",
                installer=installer,
                cumm_header_repairer=repairer,
                cumm_core_validator=validator,
                cumm_subint_header_patcher=subint_patcher,
                cumm_half_header_patcher=lambda *, venv_dir: {
                    "package": "cumm-half-header-patch",
                    "status": "patched",
                    "next_action": "none",
                },
                cumm_bfloat16_header_patcher=lambda *, venv_dir: {
                    "package": "cumm-bfloat16-header-patch",
                    "status": "patched",
                    "next_action": "none",
                },
                cumm_dtype_header_patcher=dtype_patcher,
            )

        self.assertEqual(probe_mock.call_count, 1)
        self.assertEqual(events, ["install:cumm", "repair:cumm", "validate:cumm.core_cc", "patch:cumm-subint", "patch:cumm-dtypes"])
        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["ready"])
        self.assertIn("dtype header patch failed", result["message"])
        self.assertEqual(result["next_action"], "inspect float8 header")
        self.assertEqual([entry["package"] for entry in result["install_results"]], [
            "cumm",
            "cumm-tensorview-headers",
            "cumm-core-cc",
            "cumm-subint-header-patch",
            "cumm-half-header-patch",
            "cumm-bfloat16-header-patch",
            "cumm-dtype-header-patch",
        ])

    def test_run_native_runtime_phase_stops_before_dtype_and_spconv_when_cumm_half_patch_fails(self) -> None:
        plan = self.setup.NativeRuntimePlan(
            native_mode="install",
            host=self.setup.NativeHostLane(
                system="Linux",
                machine="arm64",
                platform_key="linux/arm64",
                python_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                lane="linux-arm64-cu128",
                state="supported",
                reason="ok",
            ),
            desired_cuda_label="cu128",
            toolkit_root="/usr/local/cuda-12.8",
            nvcc_version="12.8",
            toolkit_status="matched",
            toolkit_reason="ok",
            required_imports=("spconv.pytorch", "torch_scatter", "torch_cluster"),
            related_packages=("cumm",),
            install_steps=(
                self.setup.NativeInstallStep(
                    package="cumm",
                    requirement="cumm",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "cumm"),
                ),
                self.setup.NativeInstallStep(
                    package="spconv",
                    requirement="spconv",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "spconv"),
                ),
            ),
            install_strategy="linux-arm64-source-build",
            automatic_install_supported=True,
            status="planned",
            message="install",
            next_action="install",
        )
        blocked_probe = self.setup.NativeRuntimeProbe(
            ready=False,
            status="blocked",
            managed_python="/tmp/python",
            imports={},
            missing_modules=("spconv", "cumm"),
            blocked_imports=("spconv.pytorch",),
            message="blocked",
            next_action="repair",
        )
        events: list[str] = []

        def installer(*, venv_dir: Path, step: object) -> dict[str, str]:
            events.append(f"install:{step.package}")
            return {"package": step.package, "status": "installed"}

        def repairer(*, venv_dir: Path, extension_dir: Path) -> dict[str, str]:
            events.append("repair:cumm")
            return {"package": "cumm-tensorview-headers", "status": "repaired", "next_action": "none"}

        def validator(*, managed_python: Path) -> dict[str, object]:
            events.append("validate:cumm.core_cc")
            return {
                "package": "cumm-core-cc",
                "module": "cumm.core_cc",
                "ready": True,
                "status": "ready",
                "next_action": "none",
            }

        def subint_patcher(*, venv_dir: Path) -> dict[str, object]:
            events.append("patch:cumm-subint")
            return {
                "package": "cumm-subint-header-patch",
                "status": "patched",
                "next_action": "none",
            }

        def half_patcher(*, venv_dir: Path) -> dict[str, object]:
            events.append("patch:cumm-half")
            return {
                "package": "cumm-half-header-patch",
                "status": "failed",
                "reason": "unknown_half_header_source",
                "next_action": "inspect half header",
            }

        dtype_patcher = mock.Mock()

        with (
            mock.patch.object(self.setup, "build_native_runtime_plan", return_value=plan),
            mock.patch.object(self.setup, "probe_native_runtime", return_value=blocked_probe) as probe_mock,
        ):
            result = self.setup.run_native_runtime_phase(
                venv_dir=Path("/tmp/venv"),
                managed_python=Path("/tmp/python"),
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                native_mode="install",
                installer=installer,
                cumm_header_repairer=repairer,
                cumm_core_validator=validator,
                cumm_subint_header_patcher=subint_patcher,
                cumm_half_header_patcher=half_patcher,
                cumm_dtype_header_patcher=dtype_patcher,
            )

        self.assertEqual(probe_mock.call_count, 1)
        self.assertEqual(events, ["install:cumm", "repair:cumm", "validate:cumm.core_cc", "patch:cumm-subint", "patch:cumm-half"])
        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["ready"])
        self.assertIn("half header patch failed", result["message"])
        self.assertEqual(result["next_action"], "inspect half header")
        dtype_patcher.assert_not_called()
        self.assertEqual([entry["package"] for entry in result["install_results"]], [
            "cumm",
            "cumm-tensorview-headers",
            "cumm-core-cc",
            "cumm-subint-header-patch",
            "cumm-half-header-patch",
        ])

    def test_run_native_runtime_phase_stops_before_dtype_and_spconv_when_cumm_bfloat16_patch_fails(self) -> None:
        plan = self.setup.NativeRuntimePlan(
            native_mode="install",
            host=self.setup.NativeHostLane(
                system="Linux",
                machine="arm64",
                platform_key="linux/arm64",
                python_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                lane="linux-arm64-cu128",
                state="supported",
                reason="ok",
            ),
            desired_cuda_label="cu128",
            toolkit_root="/usr/local/cuda-12.8",
            nvcc_version="12.8",
            toolkit_status="matched",
            toolkit_reason="ok",
            required_imports=("spconv.pytorch", "torch_scatter", "torch_cluster"),
            related_packages=("cumm",),
            install_steps=(
                self.setup.NativeInstallStep(
                    package="cumm",
                    requirement="cumm",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "cumm"),
                ),
                self.setup.NativeInstallStep(
                    package="spconv",
                    requirement="spconv",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "spconv"),
                ),
            ),
            install_strategy="linux-arm64-source-build",
            automatic_install_supported=True,
            status="planned",
            message="install",
            next_action="install",
        )
        blocked_probe = self.setup.NativeRuntimeProbe(
            ready=False,
            status="blocked",
            managed_python="/tmp/python",
            imports={},
            missing_modules=("spconv", "cumm"),
            blocked_imports=("spconv.pytorch",),
            message="blocked",
            next_action="repair",
        )
        events: list[str] = []

        def installer(*, venv_dir: Path, step: object) -> dict[str, str]:
            events.append(f"install:{step.package}")
            return {"package": step.package, "status": "installed"}

        def repairer(*, venv_dir: Path, extension_dir: Path) -> dict[str, str]:
            events.append("repair:cumm")
            return {"package": "cumm-tensorview-headers", "status": "repaired", "next_action": "none"}

        def validator(*, managed_python: Path) -> dict[str, object]:
            events.append("validate:cumm.core_cc")
            return {
                "package": "cumm-core-cc",
                "module": "cumm.core_cc",
                "ready": True,
                "status": "ready",
                "next_action": "none",
            }

        def subint_patcher(*, venv_dir: Path) -> dict[str, object]:
            events.append("patch:cumm-subint")
            return {
                "package": "cumm-subint-header-patch",
                "status": "patched",
                "next_action": "none",
            }

        def half_patcher(*, venv_dir: Path) -> dict[str, object]:
            events.append("patch:cumm-half")
            return {
                "package": "cumm-half-header-patch",
                "status": "patched",
                "next_action": "none",
            }

        def bfloat16_patcher(*, venv_dir: Path) -> dict[str, object]:
            events.append("patch:cumm-bfloat16")
            return {
                "package": "cumm-bfloat16-header-patch",
                "status": "failed",
                "reason": "unknown_bfloat16_header_source",
                "next_action": "inspect bfloat16 header",
            }

        dtype_patcher = mock.Mock()

        with (
            mock.patch.object(self.setup, "build_native_runtime_plan", return_value=plan),
            mock.patch.object(self.setup, "probe_native_runtime", return_value=blocked_probe) as probe_mock,
        ):
            result = self.setup.run_native_runtime_phase(
                venv_dir=Path("/tmp/venv"),
                managed_python=Path("/tmp/python"),
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                native_mode="install",
                installer=installer,
                cumm_header_repairer=repairer,
                cumm_core_validator=validator,
                cumm_subint_header_patcher=subint_patcher,
                cumm_half_header_patcher=half_patcher,
                cumm_bfloat16_header_patcher=bfloat16_patcher,
                cumm_dtype_header_patcher=dtype_patcher,
            )

        self.assertEqual(probe_mock.call_count, 1)
        self.assertEqual(events, ["install:cumm", "repair:cumm", "validate:cumm.core_cc", "patch:cumm-subint", "patch:cumm-half", "patch:cumm-bfloat16"])
        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["ready"])
        self.assertIn("bfloat16 header patch failed", result["message"])
        self.assertEqual(result["next_action"], "inspect bfloat16 header")
        dtype_patcher.assert_not_called()
        self.assertEqual([entry["package"] for entry in result["install_results"]], [
            "cumm",
            "cumm-tensorview-headers",
            "cumm-core-cc",
            "cumm-subint-header-patch",
            "cumm-half-header-patch",
            "cumm-bfloat16-header-patch",
        ])

    def test_run_native_runtime_phase_appends_spconv_core_validation_after_spconv_install(self) -> None:
        plan = self.setup.NativeRuntimePlan(
            native_mode="install",
            host=self.setup.NativeHostLane(
                system="Linux",
                machine="arm64",
                platform_key="linux/arm64",
                python_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                lane="linux-arm64-cu128",
                state="supported",
                reason="ok",
            ),
            desired_cuda_label="cu128",
            toolkit_root="/usr/local/cuda-12.8",
            nvcc_version="12.8",
            toolkit_status="matched",
            toolkit_reason="ok",
            required_imports=("spconv.pytorch", "torch_scatter", "torch_cluster"),
            related_packages=("cumm",),
            install_steps=(
                self.setup.NativeInstallStep(
                    package="spconv",
                    requirement="spconv",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "spconv"),
                ),
            ),
            install_strategy="linux-arm64-source-build",
            automatic_install_supported=True,
            status="planned",
            message="install",
            next_action="install",
        )
        blocked_probe = self.setup.NativeRuntimeProbe(
            ready=False,
            status="blocked",
            managed_python="/tmp/python",
            imports={},
            missing_modules=("spconv",),
            blocked_imports=("spconv.pytorch",),
            message="blocked",
            next_action="repair",
        )
        ready_probe = self.setup.NativeRuntimeProbe(
            ready=True,
            status="ready",
            managed_python="/tmp/python",
            imports={"spconv.pytorch": {"ready": True, "status": "ready"}},
            missing_modules=(),
            blocked_imports=(),
            message="ready",
            next_action="none",
        )

        with (
            mock.patch.object(self.setup, "build_native_runtime_plan", return_value=plan),
            mock.patch.object(self.setup, "probe_native_runtime", side_effect=[blocked_probe, ready_probe]) as probe_mock,
        ):
            installer = mock.Mock(return_value={"package": "spconv", "status": "installed"})
            spconv_validator = mock.Mock(
                return_value={
                    "package": "spconv-core-cc",
                    "module": "spconv.core_cc",
                    "ready": True,
                    "status": "ready",
                    "next_action": "none",
                }
            )
            result = self.setup.run_native_runtime_phase(
                venv_dir=Path("/tmp/venv"),
                managed_python=Path("/tmp/python"),
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                native_mode="install",
                installer=installer,
                spconv_core_validator=spconv_validator,
            )

        self.assertEqual(probe_mock.call_count, 2)
        spconv_validator.assert_called_once_with(managed_python=Path("/tmp/python"))
        self.assertEqual([entry["package"] for entry in result["install_results"]], ["spconv", "spconv-core-cc"])
        self.assertTrue(result["ready"])

    def test_run_native_runtime_phase_stops_when_spconv_core_cc_validation_fails(self) -> None:
        plan = self.setup.NativeRuntimePlan(
            native_mode="install",
            host=self.setup.NativeHostLane(
                system="Linux",
                machine="arm64",
                platform_key="linux/arm64",
                python_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                lane="linux-arm64-cu128",
                state="supported",
                reason="ok",
            ),
            desired_cuda_label="cu128",
            toolkit_root="/usr/local/cuda-12.8",
            nvcc_version="12.8",
            toolkit_status="matched",
            toolkit_reason="ok",
            required_imports=("spconv.pytorch", "torch_scatter", "torch_cluster"),
            related_packages=("cumm",),
            install_steps=(
                self.setup.NativeInstallStep(
                    package="spconv",
                    requirement="spconv",
                    strategy="pip",
                    status="planned",
                    pip_args=("install", "spconv"),
                ),
            ),
            install_strategy="linux-arm64-source-build",
            automatic_install_supported=True,
            status="planned",
            message="install",
            next_action="install",
        )
        blocked_probe = self.setup.NativeRuntimeProbe(
            ready=False,
            status="blocked",
            managed_python="/tmp/python",
            imports={},
            missing_modules=("spconv",),
            blocked_imports=("spconv.pytorch",),
            message="blocked",
            next_action="repair",
        )

        with (
            mock.patch.object(self.setup, "build_native_runtime_plan", return_value=plan),
            mock.patch.object(self.setup, "probe_native_runtime", return_value=blocked_probe) as probe_mock,
        ):
            installer = mock.Mock(return_value={"package": "spconv", "status": "installed"})
            spconv_validator = mock.Mock(
                return_value={
                    "package": "spconv-core-cc",
                    "module": "spconv.core_cc",
                    "ready": False,
                    "status": "missing",
                    "error": "No module named 'spconv.core_cc'",
                    "traceback_tail": ["ModuleNotFoundError: No module named 'spconv.core_cc'"],
                    "next_action": "Rebuild spconv with SPCONV_DISABLE_JIT=1 so spconv.core_cc is emitted during install.",
                }
            )
            result = self.setup.run_native_runtime_phase(
                venv_dir=Path("/tmp/venv"),
                managed_python=Path("/tmp/python"),
                system_name="Linux",
                machine="aarch64",
                py_tag="cp311",
                gpu_sm=100,
                cuda_version=128,
                native_mode="install",
                installer=installer,
                spconv_core_validator=spconv_validator,
            )

        self.assertEqual(probe_mock.call_count, 1)
        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["ready"])
        self.assertIn("spconv.core_cc", result["message"])
        self.assertEqual(result["next_action"], "Rebuild spconv with SPCONV_DISABLE_JIT=1 so spconv.core_cc is emitted during install.")
        self.assertEqual([entry["package"] for entry in result["install_results"]], ["spconv", "spconv-core-cc"])

    def test_install_native_runtime_step_captures_failure_details(self) -> None:
        step = self.setup.NativeInstallStep(
            package="cumm",
            requirement="git+https://github.com/FindDefinition/cumm.git@v0.7.11",
            strategy="pip-git-source-build",
            status="planned",
            pip_args=("install", "--no-build-isolation", "--no-deps", "pkg"),
            env={"FORCE_CUDA": "1", "TORCH_CUDA_ARCH_LIST": "8.9", "CUMM_CUDA_ARCH_LIST": "8.9"},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir = Path(temp_dir) / "venv"
            with mock.patch.object(self.setup.subprocess, "run") as run_mock:
                run_mock.return_value = mock.Mock(returncode=2, stdout="line1\nline2\n", stderr="err1\nerr2\n")
                result = self.setup.install_native_runtime_step(venv_dir=venv_dir, step=step)

            _, kwargs = run_mock.call_args
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["returncode"], 2)
            self.assertEqual(result["stdout_tail"], ["line1", "line2"])
            self.assertEqual(result["env"]["FORCE_CUDA"], "1")
            self.assertEqual(result["cwd"], temp_dir)
            self.assertEqual(result["command"], [str(venv_dir / "bin" / "python"), "-m", "pip", "install", "--no-build-isolation", "--no-deps", "pkg"])
            self.assertEqual(kwargs["cwd"], temp_dir)
            self.assertEqual(kwargs["env"]["TMPDIR"], str(Path(temp_dir) / ".modly" / "native-build-artifacts" / "tmp" / "cumm"))
            self.assertEqual(kwargs["env"]["PIP_NO_CLEAN"], "1")
            self.assertEqual(kwargs["env"]["PIP_VERBOSE"], "1")
            self.assertEqual(kwargs["env"]["CMAKE_VERBOSE_MAKEFILE"], "ON")
            self.assertEqual(kwargs["env"]["VERBOSE"], "1")
            self.assertEqual(Path(result["stdout_log_path"]).read_text(encoding="utf-8"), "line1\nline2\n")
            self.assertEqual(Path(result["stderr_log_path"]).read_text(encoding="utf-8"), "err1\nerr2\n")

    def test_install_native_runtime_step_extracts_compiler_and_linker_error_lines(self) -> None:
        step = self.setup.NativeInstallStep(
            package="spconv",
            requirement="git+https://github.com/traveller59/spconv.git@v2.3.8",
            strategy="pip-git-source-build",
            status="planned",
            pip_args=("install", "--no-build-isolation", "--no-deps", "pkg"),
            env={"SPCONV_DISABLE_JIT": "1"},
        )
        stdout = "\n".join(
            [
                "[1/4] c++ -Ifoo -c src.cc",
                "src.cc:10:10: fatal error: tensorview/tensor.h: No such file or directory",
                "ninja: build stopped: subcommand failed.",
            ]
        )
        stderr = "\n".join(
            [
                "collect2: error: ld returned 1 exit status",
                "subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.",
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(self.setup.subprocess, "run") as run_mock:
                run_mock.return_value = mock.Mock(returncode=1, stdout=stdout, stderr=stderr)
                result = self.setup.install_native_runtime_step(venv_dir=Path(temp_dir) / "venv", step=step)

        self.assertEqual(
            result["build_error_lines"],
            [
                "src.cc:10:10: fatal error: tensorview/tensor.h: No such file or directory",
                "ninja: build stopped: subcommand failed.",
                "collect2: error: ld returned 1 exit status",
                "subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.",
            ],
        )

    def test_summarize_native_build_artifacts_collects_install_results(self) -> None:
        extension_dir = Path("/tmp/hunyuan3d-part")
        summary = self.setup.summarize_native_build_artifacts(
            extension_dir=extension_dir,
            native_runtime={
                "plan": {
                    "desired_cuda_label": "cu128",
                    "toolkit_root": "/usr/local/cuda-12.8",
                    "toolkit_include_dir": "/usr/local/cuda-12.8/include",
                    "toolkit_nvcc": "/usr/local/cuda-12.8/bin/nvcc",
                    "nvcc_version": "12.8",
                    "toolkit_status": "matched",
                    "toolkit_reason": "ok",
                },
                "install_results": [
                    {
                        "package": "cumm-cuda-resolver-patch",
                        "status": "patched",
                        "path": str(extension_dir / "venv" / "lib" / "python3.12" / "site-packages" / "cumm" / "common.py"),
                        "action": "rewrite",
                        "selected_toolkit_root": "/usr/local/cuda-12.8",
                        "selected_include_dir": "/usr/local/cuda-12.8/include",
                        "selected_nvcc_path": "/usr/local/cuda-12.8/bin/nvcc",
                        "no_global_cuda_mutation": True,
                    },
                    {
                        "package": "spconv",
                        "status": "failed",
                        "command": ["python", "-m", "pip", "install", "spconv"],
                        "cwd": str(extension_dir),
                        "artifact_root": str(extension_dir / ".modly" / "native-build-artifacts"),
                        "artifact_tmp_dir": str(extension_dir / ".modly" / "native-build-artifacts" / "tmp" / "spconv"),
                        "build_tracker_dir": str(extension_dir / ".modly" / "native-build-artifacts" / "pip-build-tracker" / "spconv"),
                        "stdout_log_path": str(extension_dir / ".modly" / "native-build-artifacts" / "logs" / "spconv.stdout.log"),
                        "stderr_log_path": str(extension_dir / ".modly" / "native-build-artifacts" / "logs" / "spconv.stderr.log"),
                    },
                    {"package": "probe", "status": "ready"},
                ]
            },
        )

        self.assertEqual(summary["root"], str(extension_dir / ".modly" / "native-build-artifacts"))
        self.assertEqual(summary["cuda"]["toolkit_root"], "/usr/local/cuda-12.8")
        self.assertEqual(summary["cuda"]["include_dir"], "/usr/local/cuda-12.8/include")
        self.assertEqual(summary["cuda"]["nvcc_path"], "/usr/local/cuda-12.8/bin/nvcc")
        self.assertEqual(summary["cuda"]["resolver_patch"]["package"], "cumm-cuda-resolver-patch")
        self.assertEqual(summary["cuda"]["resolver_patch"]["status"], "patched")
        self.assertTrue(summary["cuda"]["resolver_patch"]["no_global_cuda_mutation"])
        self.assertEqual(len(summary["steps"]), 1)
        self.assertEqual(summary["steps"][0]["package"], "spconv")
        self.assertEqual(summary["steps"][0]["command"], ["python", "-m", "pip", "install", "spconv"])

    def test_build_error_lines_deduplicates_matching_lines(self) -> None:
        lines = self.setup._build_error_lines(
            "error: first\nerror: first\n",
            "ld: second\n",
        )

        self.assertEqual(lines, ["error: first", "ld: second"])

    def test_prepare_upstream_runtime_source_reuses_existing_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extension_dir = Path(temp_dir)
            runtime_root = self.setup.runtime_source_root(extension_dir)
            model_path = runtime_root / "P3-SAM" / "model.py"
            sonata_model_path = runtime_root / "XPart" / "partgen" / "models" / "sonata" / "model.py"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_text(
                "import os\nroot = self.sonata = sonata.load('sonata', repo_id='facebook/sonata', download_root='/root/sonata')\n",
                encoding="utf-8",
            )
            sonata_model_path.parent.mkdir(parents=True, exist_ok=True)
            sonata_model_path.write_text(
                "# 关闭flash attention\n# ckpt[\"config\"]['enable_flash'] = False\nmodel = PointTransformerV3(**ckpt[\"config\"])\n\n"
                "def load_by_config(config_path: str):\n    with open(config_path, \"r\") as f:\n        config = json.load(f)\n    model = PointTransformerV3(**config)\n    return model\n",
                encoding="utf-8",
            )
            for relative_path in self.setup.UPSTREAM_REQUIRED_PATHS[1:]:
                target = runtime_root / relative_path
                if target.suffix:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text("ok\n", encoding="utf-8")
                else:
                    target.mkdir(parents=True, exist_ok=True)

            result = self.setup.prepare_upstream_runtime_source(extension_dir)

        self.assertTrue(result["ready"])
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["patch"]["status"], "already_patched")
        self.assertEqual(result["patch"]["components"]["sonata_flash_attention_patch"]["status"], "already_patched")

    def test_patch_prepared_upstream_source_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extension_dir = Path(temp_dir)
            model_path = self.setup.runtime_source_root(extension_dir) / "P3-SAM" / "model.py"
            sonata_model_path = self.setup.runtime_source_root(extension_dir) / "XPart" / "partgen" / "models" / "sonata" / "model.py"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_text(
                "import os\nself.sonata = sonata.load(\"sonata\", repo_id=\"facebook/sonata\", download_root='/root/sonata')\n",
                encoding="utf-8",
            )
            sonata_model_path.parent.mkdir(parents=True, exist_ok=True)
            sonata_model_path.write_text(
                "# 关闭flash attention\n# ckpt[\"config\"]['enable_flash'] = False\nmodel = PointTransformerV3(**ckpt[\"config\"])\n\n"
                "def load_by_config(config_path: str):\n    with open(config_path, \"r\") as f:\n        config = json.load(f)\n    model = PointTransformerV3(**config)\n    return model\n",
                encoding="utf-8",
            )

            first = self.setup.patch_prepared_upstream_source(extension_dir)
            second = self.setup.patch_prepared_upstream_source(extension_dir)
            content = model_path.read_text(encoding="utf-8")
            sonata_content = sonata_model_path.read_text(encoding="utf-8")

        self.assertEqual(first["status"], "patched")
        self.assertEqual(second["status"], "already_patched")
        self.assertIn(self.setup.SONATA_CACHE_ENV_VAR, content)
        self.assertNotIn("/root/sonata", content)
        self.assertIn(self.setup.SONATA_FLASH_ATTENTION_LOAD_PATCH_NEW, sonata_content)
        self.assertIn(self.setup.SONATA_FLASH_ATTENTION_LOAD_BY_CONFIG_PATCH_NEW, sonata_content)
        self.assertEqual(first["components"]["p3_sam_sonata_cache_patch"]["status"], "patched")
        self.assertEqual(first["components"]["sonata_flash_attention_patch"]["status"], "patched")
        self.assertEqual(first["components"]["sonata_flash_attention_patch"]["sites"]["load"]["status"], "patched")
        self.assertEqual(first["components"]["sonata_flash_attention_patch"]["sites"]["load_by_config"]["status"], "patched")
        self.assertEqual(second["components"]["sonata_flash_attention_patch"]["status"], "already_patched")

    def test_patch_prepared_upstream_source_inserts_sonata_flash_fallback_before_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extension_dir = Path(temp_dir)
            runtime_root = self.setup.runtime_source_root(extension_dir)
            p3_model_path = runtime_root / "P3-SAM" / "model.py"
            sonata_model_path = runtime_root / "XPart" / "partgen" / "models" / "sonata" / "model.py"
            p3_model_path.parent.mkdir(parents=True, exist_ok=True)
            p3_model_path.write_text(
                "import os\nself.sonata = sonata.load('sonata', repo_id='facebook/sonata', download_root='/root/sonata')\n",
                encoding="utf-8",
            )
            sonata_model_path.parent.mkdir(parents=True, exist_ok=True)
            sonata_model_path.write_text(
                "def build(ckpt):\n    # 关闭flash attention\n    # ckpt[\"config\"]['enable_flash'] = False\n\n    model = PointTransformerV3(**ckpt[\"config\"])\n    return model\n\n"
                "def load_by_config(config_path: str):\n    with open(config_path, \"r\") as f:\n        config = json.load(f)\n    model = PointTransformerV3(**config)\n    return model\n",
                encoding="utf-8",
            )

            patch = self.setup.patch_prepared_upstream_source(extension_dir)
            content = sonata_model_path.read_text(encoding="utf-8")

        self.assertTrue(patch["ready"])
        self.assertEqual(patch["components"]["sonata_flash_attention_patch"]["status"], "patched")
        self.assertIn(
            '    ckpt["config"]["enable_flash"] = False\n\n    model = PointTransformerV3(**ckpt["config"])',
            content,
        )
        self.assertIn(
            '    config["enable_flash"] = False\n    model = PointTransformerV3(**config)',
            content,
        )
        self.assertLess(
            content.index('    config["enable_flash"] = False'),
            content.index('    model = PointTransformerV3(**config)'),
        )

    def test_patch_prepared_upstream_source_reports_already_patched_sonata_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extension_dir = Path(temp_dir)
            runtime_root = self.setup.runtime_source_root(extension_dir)
            p3_model_path = runtime_root / "P3-SAM" / "model.py"
            sonata_model_path = runtime_root / "XPart" / "partgen" / "models" / "sonata" / "model.py"
            p3_model_path.parent.mkdir(parents=True, exist_ok=True)
            p3_model_path.write_text(
                f"import os\nself.sonata = sonata.load('sonata', repo_id='facebook/sonata', {self.setup.UPSTREAM_MODEL_PATCH_NEW})\n",
                encoding="utf-8",
            )
            sonata_model_path.parent.mkdir(parents=True, exist_ok=True)
            sonata_model_path.write_text(
                "def build(ckpt):\n    # 关闭flash attention\n    ckpt[\"config\"][\"enable_flash\"] = False\n\n    model = PointTransformerV3(**ckpt[\"config\"])\n    return model\n\n"
                "def load_by_config(config_path: str):\n    with open(config_path, \"r\") as f:\n        config = json.load(f)\n    config[\"enable_flash\"] = False\n    model = PointTransformerV3(**config)\n    return model\n",
                encoding="utf-8",
            )

            patch = self.setup.patch_prepared_upstream_source(extension_dir)

        self.assertEqual(patch["status"], "already_patched")
        self.assertEqual(patch["components"]["sonata_flash_attention_patch"]["status"], "already_patched")
        self.assertEqual(patch["components"]["sonata_flash_attention_patch"]["sites"]["load"]["status"], "already_patched")
        self.assertEqual(patch["components"]["sonata_flash_attention_patch"]["sites"]["load_by_config"]["status"], "already_patched")

    def test_patch_prepared_upstream_source_fails_closed_for_unknown_sonata_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extension_dir = Path(temp_dir)
            runtime_root = self.setup.runtime_source_root(extension_dir)
            p3_model_path = runtime_root / "P3-SAM" / "model.py"
            sonata_model_path = runtime_root / "XPart" / "partgen" / "models" / "sonata" / "model.py"
            p3_model_path.parent.mkdir(parents=True, exist_ok=True)
            p3_model_path.write_text(
                "import os\nself.sonata = sonata.load('sonata', repo_id='facebook/sonata', download_root='/root/sonata')\n",
                encoding="utf-8",
            )
            sonata_model_path.parent.mkdir(parents=True, exist_ok=True)
            sonata_model_path.write_text(
                "def build(ckpt):\n    flash_enabled = ckpt[\"config\"].get(\"enable_flash\", True)\n    model = PointTransformerV3(**ckpt[\"config\"])\n    return flash_enabled, model\n\n"
                "def load_by_config(config_path: str):\n    with open(config_path, \"r\") as f:\n        config = json.load(f)\n    model = PointTransformerV3(**config)\n    return model\n",
                encoding="utf-8",
            )

            patch = self.setup.patch_prepared_upstream_source(extension_dir)

        self.assertFalse(patch["ready"])
        self.assertEqual(patch["status"], "invalid")
        self.assertEqual(
            patch["components"]["sonata_flash_attention_patch"]["reason"],
            "unexpected_sonata_flash_attention_patch_shape",
        )
        self.assertEqual(patch["components"]["sonata_flash_attention_patch"]["sites"]["load_by_config"]["status"], "patched")

    def test_patch_prepared_upstream_source_fails_closed_for_unknown_load_by_config_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extension_dir = Path(temp_dir)
            runtime_root = self.setup.runtime_source_root(extension_dir)
            p3_model_path = runtime_root / "P3-SAM" / "model.py"
            sonata_model_path = runtime_root / "XPart" / "partgen" / "models" / "sonata" / "model.py"
            p3_model_path.parent.mkdir(parents=True, exist_ok=True)
            p3_model_path.write_text(
                "import os\nself.sonata = sonata.load('sonata', repo_id='facebook/sonata', download_root='/root/sonata')\n",
                encoding="utf-8",
            )
            sonata_model_path.parent.mkdir(parents=True, exist_ok=True)
            sonata_model_path.write_text(
                "def build(ckpt):\n    # 关闭flash attention\n    # ckpt[\"config\"]['enable_flash'] = False\n\n    model = PointTransformerV3(**ckpt[\"config\"])\n    return model\n\n"
                "def load_by_config(config_path: str):\n    with open(config_path, \"r\") as f:\n        config = json.load(f)\n    config.setdefault(\"enable_flash\", True)\n    model = PointTransformerV3(**config)\n    return model\n",
                encoding="utf-8",
            )

            patch = self.setup.patch_prepared_upstream_source(extension_dir)

        self.assertFalse(patch["ready"])
        self.assertEqual(patch["status"], "invalid")
        self.assertEqual(
            patch["components"]["sonata_flash_attention_patch"]["sites"]["load_by_config"]["reason"],
            "unexpected_sonata_flash_attention_patch_shape",
        )

    def test_prepare_upstream_runtime_source_clones_sparse_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extension_dir = Path(temp_dir)

            def fake_run(command, check):
                self.assertTrue(check)
                runtime_root = self.setup.runtime_source_root(extension_dir)
                if command[:2] == ["git", "clone"]:
                    runtime_root.mkdir(parents=True, exist_ok=True)
                elif command[:4] == ["git", "-C", str(runtime_root), "sparse-checkout"]:
                    model_path = runtime_root / "P3-SAM" / "model.py"
                    sonata_model_path = runtime_root / "XPart" / "partgen" / "models" / "sonata" / "model.py"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    model_path.write_text(
                        "import os\nself.sonata = sonata.load('sonata', repo_id='facebook/sonata', download_root='/root/sonata')\n",
                        encoding="utf-8",
                    )
                    sonata_model_path.parent.mkdir(parents=True, exist_ok=True)
                    sonata_model_path.write_text(
                        "# 关闭flash attention\n# ckpt[\"config\"]['enable_flash'] = False\nmodel = PointTransformerV3(**ckpt[\"config\"])\n\n"
                        "def load_by_config(config_path: str):\n    with open(config_path, \"r\") as f:\n        config = json.load(f)\n    model = PointTransformerV3(**config)\n    return model\n",
                        encoding="utf-8",
                    )
                    for relative_path in self.setup.UPSTREAM_REQUIRED_PATHS[1:]:
                        target = runtime_root / relative_path
                        if target.suffix:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_text("ok\n", encoding="utf-8")
                        else:
                            target.mkdir(parents=True, exist_ok=True)
                return mock.Mock()

            with mock.patch.object(self.setup.subprocess, "run", side_effect=fake_run) as run_mock:
                result = self.setup.prepare_upstream_runtime_source(extension_dir)

        self.assertTrue(result["ready"])
        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(result["patch"]["status"], "already_patched")
        self.assertEqual(result["patch"]["components"]["sonata_flash_attention_patch"]["status"], "already_patched")

    def test_run_setup_writes_ready_summary_without_real_install(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extension_dir = Path(temp_dir)
            readiness = {
                "setup_contract": "python-root-setup-py",
                "surface_owner": "electron",
                "status": "ready",
                "native_runtime_ready": True,
                "weights_ready": True,
                "adapter_ready": True,
                "runtime_ready": True,
                "execution_ready": True,
                "inference_ready": False,
                "venv": {"python": str(extension_dir / "venv" / "bin" / "python")},
                "base_runtime": {"status": "ready"},
                "runtime_adapter": {"ready": True, "status": "ready"},
                "adapter_import_smoke": {"status": "ready"},
                "phases": {"native_runtime": {"status": "ready"}},
                "native_runtime": {"mode": "auto", "status": "ready"},
                "native_dependency_policy": {"strategy": "planned"},
                "native_dependency_blockers": {"status": "clear"},
                "blocked_reasons": [],
                "next_action": "Provide runtime adapter and weights before real inference.",
            }
            with (
                mock.patch.object(self.setup.subprocess, "run") as run_mock,
                mock.patch.object(self.setup, "pip_install") as pip_install_mock,
                mock.patch.object(self.setup, "python_tag", return_value="cp311"),
                mock.patch.object(self.setup, "_platform_facts", return_value=("Linux", "aarch64")),
                mock.patch.object(self.setup, "prepare_upstream_runtime_source", return_value={"ready": True, "status": "prepared"}),
                mock.patch.object(
                    self.setup,
                    "run_native_runtime_phase",
                    return_value={
                        "mode": "auto",
                        "status": "blocked",
                        "ready": False,
                        "install_strategy": "linux-arm64-source-build",
                        "plan": {
                            "desired_cuda_label": "cu128",
                            "toolkit_root": "/usr/local/cuda-12.8",
                            "toolkit_include_dir": "/usr/local/cuda-12.8/include",
                            "toolkit_nvcc": "/usr/local/cuda-12.8/bin/nvcc",
                            "nvcc_version": "12.8",
                            "toolkit_status": "matched",
                            "toolkit_reason": "ok",
                        },
                        "install_results": [
                            {
                                "package": "cumm-cuda-resolver-patch",
                                "status": "patched",
                                "path": str(extension_dir / "venv" / "lib" / "python3.12" / "site-packages" / "cumm" / "common.py"),
                                "action": "rewrite",
                                "selected_toolkit_root": "/usr/local/cuda-12.8",
                                "selected_include_dir": "/usr/local/cuda-12.8/include",
                                "selected_nvcc_path": "/usr/local/cuda-12.8/bin/nvcc",
                                "no_global_cuda_mutation": True,
                            },
                            {
                                "package": "cumm-half-header-patch",
                                "status": "patched",
                                "path": str(extension_dir / "venv" / "lib" / "python3.12" / "site-packages" / "cumm" / "include" / "tensorview" / "gemm" / "dtypes" / "half.h"),
                            },
                            {
                                "package": "cumm-bfloat16-header-patch",
                                "status": "patched",
                                "path": str(extension_dir / "venv" / "lib" / "python3.12" / "site-packages" / "cumm" / "include" / "tensorview" / "gemm" / "dtypes" / "bfloat16.h"),
                            },
                            {
                                "package": "spconv",
                                "status": "failed",
                                "command": ["python", "-m", "pip", "install", "spconv"],
                                "cwd": str(extension_dir),
                                "artifact_root": str(extension_dir / ".modly" / "native-build-artifacts"),
                                "artifact_tmp_dir": str(extension_dir / ".modly" / "native-build-artifacts" / "tmp" / "spconv"),
                                "build_tracker_dir": str(extension_dir / ".modly" / "native-build-artifacts" / "pip-build-tracker" / "spconv"),
                                "stdout_log_path": str(extension_dir / ".modly" / "native-build-artifacts" / "logs" / "spconv.stdout.log"),
                                "stderr_log_path": str(extension_dir / ".modly" / "native-build-artifacts" / "logs" / "spconv.stderr.log"),
                            }
                        ],
                    },
                ),
                mock.patch.object(self.setup, "build_readiness_report", return_value=readiness),
            ):
                exit_code = self.setup.run_setup(
                    [json.dumps({"python_exe": "/usr/bin/python3", "ext_dir": str(extension_dir), "gpu_sm": 100, "cuda_version": 128})],
                    as_json=False,
                )
                summary = json.loads((extension_dir / ".modly-setup-summary.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(run_mock.call_count, 1)
        self.assertGreaterEqual(pip_install_mock.call_count, 3)
        self.assertEqual(summary["status"], "ready")
        self.assertEqual(summary["torch_target"], "linux-arm64-cu128")
        self.assertEqual(summary["native_mode"], "auto")
        self.assertEqual(summary["install_strategy"], "linux-arm64-source-build")
        self.assertEqual(summary["native_dependency_policy"]["strategy"], "planned")
        self.assertEqual(summary["native_dependency_blockers"]["status"], "clear")
        self.assertEqual(summary["native_build_artifacts"]["root"], str(extension_dir / ".modly" / "native-build-artifacts"))
        self.assertEqual(summary["native_build_artifacts"]["cuda"]["toolkit_root"], "/usr/local/cuda-12.8")
        self.assertEqual(summary["native_build_artifacts"]["cuda"]["include_dir"], "/usr/local/cuda-12.8/include")
        self.assertEqual(summary["native_build_artifacts"]["cuda"]["nvcc_path"], "/usr/local/cuda-12.8/bin/nvcc")
        self.assertEqual(summary["native_build_artifacts"]["cuda"]["resolver_patch"]["status"], "patched")
        self.assertEqual(summary["native_build_artifacts"]["steps"][0]["package"], "spconv")
        self.assertEqual(summary["native_runtime"]["install_results"][0]["package"], "cumm-cuda-resolver-patch")
        self.assertEqual(summary["native_runtime"]["install_results"][1]["package"], "cumm-half-header-patch")
        self.assertEqual(summary["native_runtime"]["install_results"][2]["package"], "cumm-bfloat16-header-patch")
        self.assertIn("scikit-image", summary["runtime_packages"])
        self.assertIn("diffusers", summary["runtime_packages"])
        self.assertIn("einops", summary["runtime_packages"])
        self.assertIn("pymeshlab", summary["runtime_packages"])

    def test_run_setup_summary_keeps_native_success_when_only_weights_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extension_dir = Path(temp_dir)
            weight_path = extension_dir / ".runtime" / "models" / "p3sam" / "p3sam.safetensors"
            readiness = {
                "setup_contract": "python-root-setup-py",
                "surface_owner": "electron",
                "status": "pending",
                "native_runtime_ready": True,
                "weights_ready": False,
                "adapter_ready": False,
                "runtime_ready": False,
                "execution_ready": False,
                "inference_ready": False,
                "venv": {"python": str(extension_dir / "venv" / "bin" / "python")},
                "base_runtime": {"ready": True, "status": "ready"},
                "upstream_runtime": {"ready": True, "status": "prepared"},
                "runtime_adapter": {
                    "ready": False,
                    "status": "blocked",
                    "components": {
                        "managed_python": {"ready": True, "status": "ready"},
                        "runtime_source": {"ready": True, "status": "ready"},
                        "entrypoint": {"ready": True, "status": "ready"},
                        "import_smoke": {"ready": True, "status": "ready"},
                        "weights": {"ready": False, "status": "missing", "path": str(weight_path)},
                    },
                },
                "adapter_import_smoke": {"status": "ready", "ready": True, "blocked_by_native_runtime": False},
                "phases": {"native_runtime": {"status": "ready"}, "weights": {"status": "missing"}},
                "native_runtime": {"mode": "install", "status": "ready", "ready": True},
                "weights": {"ready": False, "status": "missing", "path": str(weight_path)},
                "support": {"ready": True, "status": "ready"},
                "native_dependency_policy": {"strategy": "planned"},
                "native_dependency_blockers": {"status": "clear"},
                "blocked_reasons": [
                    "Missing required model weights: " + str(weight_path)
                ],
                "next_action": "Download weights before inference.",
            }
            with (
                mock.patch.object(self.setup.subprocess, "run") as run_mock,
                mock.patch.object(self.setup, "pip_install") as pip_install_mock,
                mock.patch.object(self.setup, "python_tag", return_value="cp311"),
                mock.patch.object(self.setup, "_platform_facts", return_value=("Linux", "aarch64")),
                mock.patch.object(self.setup, "prepare_upstream_runtime_source", return_value={"ready": True, "status": "prepared"}),
                mock.patch.object(
                    self.setup,
                    "run_native_runtime_phase",
                    return_value={
                        "mode": "install",
                        "status": "ready",
                        "ready": True,
                        "install_strategy": "linux-arm64-source-build",
                        "plan": {},
                        "install_results": [],
                    },
                ),
                mock.patch.object(self.setup, "build_readiness_report", return_value=readiness),
            ):
                exit_code = self.setup.run_setup(
                    [json.dumps({"python_exe": "/usr/bin/python3", "ext_dir": str(extension_dir), "gpu_sm": 100, "cuda_version": 128})],
                    as_json=False,
                )
                summary = json.loads((extension_dir / ".modly-setup-summary.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(run_mock.call_count, 1)
        self.assertGreaterEqual(pip_install_mock.call_count, 3)
        self.assertEqual(summary["status"], "pending")
        self.assertTrue(summary["native_runtime_ready"])
        self.assertFalse(summary["weights_ready"])
        self.assertFalse(summary["execution_ready"])
        self.assertEqual(summary["native_dependency_blockers"]["status"], "clear")
        self.assertNotIn("Runtime imports failed after managed setup.", summary["blocked_reasons"])
        self.assertTrue(any("Missing required model weights:" in reason for reason in summary["blocked_reasons"]))

    def test_run_setup_still_fails_when_native_runtime_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            extension_dir = Path(temp_dir)
            readiness = {
                "setup_contract": "python-root-setup-py",
                "surface_owner": "electron",
                "status": "pending",
                "native_runtime_ready": False,
                "weights_ready": False,
                "adapter_ready": False,
                "runtime_ready": False,
                "execution_ready": False,
                "inference_ready": False,
                "venv": {"python": str(extension_dir / "venv" / "bin" / "python")},
                "base_runtime": {"ready": True, "status": "ready"},
                "upstream_runtime": {"ready": True, "status": "prepared"},
                "runtime_adapter": {
                    "ready": False,
                    "status": "blocked",
                    "components": {
                        "managed_python": {"ready": True, "status": "ready"},
                        "runtime_source": {"ready": True, "status": "ready"},
                        "entrypoint": {"ready": True, "status": "ready"},
                        "import_smoke": {"ready": False, "status": "blocked"},
                        "weights": {"ready": False, "status": "missing"},
                    },
                },
                "adapter_import_smoke": {"status": "blocked", "ready": False, "blocked_by_native_runtime": True},
                "phases": {"native_runtime": {"status": "blocked"}, "weights": {"status": "missing"}},
                "native_runtime": {
                    "mode": "install",
                    "status": "blocked",
                    "ready": False,
                    "message": "Native runtime dependencies are not ready yet.",
                },
                "weights": {"ready": False, "status": "missing"},
                "support": {"ready": True, "status": "ready"},
                "native_dependency_policy": {"strategy": "planned"},
                "native_dependency_blockers": {"status": "blocked"},
                "blocked_reasons": ["Native runtime dependencies are not ready yet."],
                "next_action": "Repair native dependencies before inference.",
            }
            with (
                mock.patch.object(self.setup.subprocess, "run") as run_mock,
                mock.patch.object(self.setup, "pip_install") as pip_install_mock,
                mock.patch.object(self.setup, "python_tag", return_value="cp311"),
                mock.patch.object(self.setup, "_platform_facts", return_value=("Linux", "aarch64")),
                mock.patch.object(self.setup, "prepare_upstream_runtime_source", return_value={"ready": True, "status": "prepared"}),
                mock.patch.object(
                    self.setup,
                    "run_native_runtime_phase",
                    return_value={
                        "mode": "install",
                        "status": "blocked",
                        "ready": False,
                        "install_strategy": "linux-arm64-source-build",
                        "plan": {},
                        "install_results": [],
                    },
                ),
                mock.patch.object(self.setup, "build_readiness_report", return_value=readiness),
            ):
                exit_code = self.setup.run_setup(
                    [json.dumps({"python_exe": "/usr/bin/python3", "ext_dir": str(extension_dir), "gpu_sm": 100, "cuda_version": 128})],
                    as_json=False,
                )

        self.assertEqual(exit_code, 1)
        self.assertGreaterEqual(run_mock.call_count, 1)
        self.assertGreaterEqual(pip_install_mock.call_count, 3)

    def test_setup_exit_code_allows_pending_weights_only_after_native_setup_succeeds(self) -> None:
        readiness = {
            "execution_ready": False,
            "base_runtime": {"ready": True},
            "upstream_runtime": {"ready": True},
            "native_runtime": {"ready": True},
            "runtime_adapter": {
                "components": {
                    "managed_python": {"ready": True},
                    "runtime_source": {"ready": True},
                    "entrypoint": {"ready": True},
                    "import_smoke": {"ready": True},
                    "weights": {"ready": False, "status": "missing"},
                }
            },
            "weights": {"ready": False},
            "support": {"status": "ready"},
        }

        self.assertEqual(self.setup._setup_exit_code(readiness), 0)


if __name__ == "__main__":
    unittest.main()
