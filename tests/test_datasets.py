"""Benchmark loaders: stable question ids and argument validation.

These tests need the benchmark question files and skip when they are absent.
"""
import pytest


def test_force3d_ids_are_stable_across_slices():
    """A question's id is its position in the full file, so every category/size slice gives it the same id."""
    from saturn.datasets.force3d import Force3DRefDataset, _DEFAULT_REF_JSON
    if not _DEFAULT_REF_JSON.exists():
        pytest.skip("needs the 3D-FORCE question files (setup/30_datasets.sh)")
    full = Force3DRefDataset(load_image=False)
    key = lambda it: (it["query"], tuple(it["image_file_name"]), str(it["answer"]))
    items = [full[i] for i in range(len(full))]
    assert len({it["id"] for it in items}) == len(items)            # ids unique
    ids_full = {key(it): it["id"] for it in items}
    sub = Force3DRefDataset(category="chain", num_samples=25, load_image=False)
    for i in range(len(sub)):
        it = sub[i]; assert ids_full[key(it)] == it["id"]          # same question -> same id in every slice
    # full-run ids are the global question numbering
    assert full[0]["id"].endswith("_q0") and full[len(full) - 1]["id"].endswith(f"_q{len(full) - 1}")


def test_mmsi_unknown_bin_raises():
    """An unknown MMSI question type is an error."""
    from saturn.datasets.mmsi import MMSIDataset, _DEFAULT_PARQUET
    if not _DEFAULT_PARQUET.exists():
        pytest.skip("needs the MMSI-Bench parquet (setup/30_datasets.sh)")
    with pytest.raises(ValueError):
        MMSIDataset(question_type="camcam", load_image=False)
