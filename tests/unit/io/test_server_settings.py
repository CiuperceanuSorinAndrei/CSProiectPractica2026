from src.io.server_settings import ServerSettings


def test_server_settings_defaults_invalid_time_delta_to_15():
    for invalid in (None, 0, -5, "bad"):
        settings = ServerSettings.from_inputs(
            host="", remote_dir="", local_dir="", file_format="",
            time_delta=invalid, username="", password="",
        )

        assert settings.time_delta == 15


def test_server_settings_accepts_positive_time_delta():
    settings = ServerSettings.from_inputs(
        host="", remote_dir="", local_dir="", file_format="",
        time_delta=30, username="", password="",
    )

    assert settings.time_delta == 30
