import os
import requests
import unittest
import numpy as np
import tempfile
from H5CosmoKit import preview
from unittest.mock import patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # This sets a non-GUI backend

class TestH5CosmoKit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.snapshot_url = "https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/CV/CV_0/snapshot_090.hdf5"
        cls.local_snapshot_path = "test/snapshot_090.hdf5"  # Path where the test HDF5 file is saved

        # Download the file if it doesn't exist
        if not os.path.exists(cls.local_snapshot_path):
            with requests.get(cls.snapshot_url, stream=True) as r:
                r.raise_for_status()
                with open(cls.local_snapshot_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

    def test_preview_rho_g(self):
        # Get the directory where the HDF5 file is located
        directory_path = os.path.dirname(self.local_snapshot_path)
        snapshot_numbers = [90]

        with patch('matplotlib.pyplot.show') as mock_show:
            preview(directory_path, snapshot_numbers, 'gas_density')
        
        mock_show.assert_called_once()

    def test_preview_temperature(self):
        # Get the directory where the HDF5 file is located
        directory_path = os.path.dirname(self.local_snapshot_path)
        snapshot_numbers = [90]

        with patch('matplotlib.pyplot.show') as mock_show:
            preview(directory_path, snapshot_numbers, 'gas_temperature')
        
        mock_show.assert_called_once()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.local_snapshot_path):
            os.remove(cls.local_snapshot_path)

if __name__ == '__main__':
    unittest.main()
