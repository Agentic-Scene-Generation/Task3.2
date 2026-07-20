from scenesmith.scenebenchmark_critic.asset_library_annotations import (
    get_hssd_asset_annotations,
)


def test_bundled_hssd_physics_and_quality_record():
    record = get_hssd_asset_annotations("0001fb06b075a743e6289236cf049df3ad5dfa9c")
    assert record is not None
    physics = record["asset_physics"]
    quality = record["asset_quality"]
    assert physics["material"] == "steel"
    assert (
        physics["mass_range_kg"][0] <= physics["mass_kg"] <= physics["mass_range_kg"][1]
    )
    assert 0.0 <= physics["friction_coefficient"] <= 2.0
    assert quality["is_acceptable"] is True
    assert quality["watertight"] is None
    assert record["scenebenchmark_fd_sa"]
