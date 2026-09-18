"""Packing a depth frame small enough to ship, without changing what it says."""

import gzip
import io

import numpy as np
import pytest

from cell import capture


def _frame(seed: int = 0) -> np.ndarray:
    """A depth frame shaped like a real one.

    Two flat faces meeting, a little sensor noise on top, the far wall behind and
    a band of pixels that returned nothing. Uniform noise would be the wrong
    fixture here: real depth is mostly smooth surface, which is exactly why it
    compresses, and a frame of pure noise would test a property the format is not
    claiming.
    """
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.8, 1.6, 64, dtype=np.float32)
    depth = np.repeat(ramp[None, :], 64, axis=0).copy()
    depth[:, 32:] = 1.2                                   # the second face
    depth += rng.normal(0.0, 0.0008, depth.shape).astype(np.float32)
    depth[:8, :] = 25.0          # the far wall, past any useful range
    depth[8:12, :] = np.nan      # no return, which every depth sensor produces
    return depth.astype(np.float32)


class TestPackingKeepsTheMeasurement:
    def test_near_field_survives_within_a_tenth_of_a_millimetre(self, tmp_path):
        depth = _frame()
        capture.pack_depth(depth, tmp_path / "f.npy.gz")
        back = capture.load_depth(tmp_path / "f.npy.gz")
        near = np.isfinite(depth) & (depth <= capture.RANGE_M)
        worst_mm = float(np.abs(back[near] - depth[near]).max()) * 1000.0
        # The tracked seam is accurate to 0.68 mm and the sensor noise floor is
        # above that, so a tenth of a millimetre here is not the binding term.
        assert worst_mm < 0.1, f"quantisation moved a pixel by {worst_mm:.4f} mm"

    def test_it_is_actually_smaller(self, tmp_path):
        depth = _frame()
        written = capture.pack_depth(depth, tmp_path / "f.npy.gz")
        assert written < depth.nbytes / 3

    def test_no_return_stays_no_return(self, tmp_path):
        depth = _frame()
        capture.pack_depth(depth, tmp_path / "f.npy.gz")
        back = capture.load_depth(tmp_path / "f.npy.gz")
        assert np.isnan(back[8:12, :]).all()

    def test_past_the_range_limit_reads_as_no_return_rather_than_as_surface(self, tmp_path):
        """The far wall must not come back as a plane the tracker could fit.

        Clamping it to the range limit instead would put a solid surface at 4 m
        across the top of every frame, which is worse than admitting no return.
        """
        depth = _frame()
        capture.pack_depth(depth, tmp_path / "f.npy.gz")
        back = capture.load_depth(tmp_path / "f.npy.gz")
        assert np.isnan(back[:8, :]).all()


class TestLoadingEitherForm:
    def test_a_plain_npy_still_loads(self, tmp_path):
        depth = _frame()
        np.save(tmp_path / "f.npy", depth)
        assert np.allclose(capture.load_depth(tmp_path / "f.npy"), depth, equal_nan=True)

    def test_a_manifest_naming_the_raw_frame_finds_the_packed_one(self, tmp_path):
        """One manifest serves a full capture and the shipped demo."""
        depth = _frame()
        capture.pack_depth(depth, tmp_path / "f.npy.gz")
        back = capture.load_depth(tmp_path / "f.npy")     # the raw name, absent
        assert back.shape == depth.shape

    def test_a_missing_frame_says_what_to_run(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="demo"):
            capture.load_depth(tmp_path / "nothing.npy")

    def test_unpack_reverses_pack(self, tmp_path):
        depth = _frame(3)
        capture.pack_depth(depth, tmp_path / "f.npy.gz")
        from_file = capture.load_depth(tmp_path / "f.npy.gz")
        from_bytes = capture.unpack_depth((tmp_path / "f.npy.gz").read_bytes())
        assert np.allclose(from_file, from_bytes, equal_nan=True)

    def test_the_packed_form_is_gzipped_numpy(self, tmp_path):
        """Nothing exotic: anyone can open it without this repository."""
        capture.pack_depth(_frame(), tmp_path / "f.npy.gz")
        raw = gzip.decompress((tmp_path / "f.npy.gz").read_bytes())
        assert np.load(io.BytesIO(raw)).dtype == np.uint16


class TestTheShippedCaptureIsUsable:
    def test_the_demo_manifest_points_at_frames_that_exist(self):
        import json
        from pathlib import Path

        manifest = Path("data/demo/manifest.json")
        if not manifest.exists():
            pytest.skip("demo capture not present")
        spec = json.loads(manifest.read_text(encoding="utf-8"))
        root = manifest.parent
        for row in spec["rows"]:
            for view in row["views"]:
                assert (root / view["depth"]).exists(), view["depth"]
        assert spec["rows"], "a demo capture with no parts in it is not a demo"
