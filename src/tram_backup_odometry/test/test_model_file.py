from tram_backup_odometry import model_file


def test_model_has_both_vehicles_and_both_directions(model):
    assert set(model.vehicles) == {"30618", "30639"}
    assert set(model.lines) == {0, 1}
    for line in model.lines.values():
        assert len(line.s) == len(line.grade) == len(line.curvature)
        assert len(line.geometry_s) == len(line.easting) == len(line.northing) == len(line.altitude)


def test_dump_and_load_keep_every_number(model, tmp_path):
    path = tmp_path / "model.json"
    model_file.dump(model, path, source="test")
    again = model_file.load(path)
    assert again.vehicles == model.vehicles
    assert again.filter == model.filter and again.wheel_check == model.wheel_check and again.shadow == model.shadow
    for d in model.lines:
        assert (again.lines[d].easting == model.lines[d].easting).all()
