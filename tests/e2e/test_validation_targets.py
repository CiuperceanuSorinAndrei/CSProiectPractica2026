from scripts.run_simulations import TARGET_RESERVOIRS, build_target_locations


def test_validation_targets_resolve_local_polygons():
    locations = build_target_locations()

    by_name = {loc["name"]: loc for loc in locations}
    assert set(by_name) == {"Craiova", *TARGET_RESERVOIRS}
    assert by_name["Craiova"]["polygon"] is None
    for name in TARGET_RESERVOIRS:
        assert by_name[name]["polygon"] is not None
        assert by_name[name]["polygon"].is_valid
