import pytest

from src.io.cloud_data_service import CloudDataService
from src.io.server_settings import ServerSettings


class FakeFtpClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.connected = False
        self.disconnected = False

    def connect(self):
        self.connected = True

    def fetch_files(self, _target_files):
        if self.fail:
            raise RuntimeError("fetch failed")
        return ["frame.nc"]

    def disconnect(self):
        self.disconnected = True


def _service_with_fake_client(fake):
    settings = ServerSettings(
        host="host",
        remote_dir="remote",
        local_dir="data/raw",
        file_format="h60_%Y%m%d_%H%M_fdk.nc.gz",
        time_delta=15,
        username="",
        password="",
    )
    service = CloudDataService(settings)
    service._ftp_client = fake
    return service


def test_download_files_disconnects_after_success():
    fake = FakeFtpClient()
    service = _service_with_fake_client(fake)

    assert service.download_files(["frame.nc.gz"]) == ["frame.nc"]
    assert fake.connected is True
    assert fake.disconnected is True


def test_download_files_disconnects_after_fetch_failure():
    fake = FakeFtpClient(fail=True)
    service = _service_with_fake_client(fake)

    with pytest.raises(RuntimeError):
        service.download_files(["frame.nc.gz"])

    assert fake.disconnected is True
