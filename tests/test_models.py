import pytest
import os
from PIL import Image
from local_model import query_local
from remote_model import query_remote, client


def test_local_model():
    img = Image.new('RGB', (100, 100), color='red')
    result = query_local(img, "test!")

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.skipif(os.getenv("HF_TOKEN") is None, reason="skipping remote model test no HF_TOKEN")
def test_remote_model():
    img = Image.new('RGB', (100, 100), color='blue')
    result = query_remote(img, "test?", client)

    assert isinstance(result, str)
    assert len(result) > 0
