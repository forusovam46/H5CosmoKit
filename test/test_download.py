import pytest
import requests
from unittest.mock import patch, mock_open
from H5CosmoKit import download_file
import io
import requests
from unittest.mock import patch, MagicMock

def mock_requests_get(*args, **kwargs):
    # Create a mock response object
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = iter([b'dummy data'])
    mock_response.raw = io.BytesIO(b'dummy data')
    mock_response.raise_for_status.return_value = None
    return mock_response

def test_successful_download():
    """
    Test successful file download.
    """
    url = "http://example.com/somefile.txt"
    local_filename = "localfile.txt"

    with patch('requests.get', side_effect=mock_requests_get):
        with patch("builtins.open", mock_open()):
            result = download_file(url, local_filename)
            assert result == local_filename

def test_http_error():
    """
    Test HTTP error handling.
    """
    url = "http://example.com/nonexistentfile.txt"
    local_filename = "localfile.txt"

    # Mock requests.get to raise HTTPError
    with patch('requests.get', side_effect=requests.HTTPError("404 Client Error")):
        with pytest.raises(requests.HTTPError):
            download_file(url, local_filename)

def test_invalid_url():
    """
    Test behavior with invalid URL.
    """
    url = "http://example.invalidurl"
    local_filename = "localfile.txt"

    # Assuming requests.get raises a ConnectionError for an invalid URL
    with patch('requests.get', side_effect=requests.ConnectionError):
        with pytest.raises(requests.ConnectionError):
            download_file(url, local_filename)
