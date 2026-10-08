import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('release',Path(__file__).parents[1]/'scripts/export_embedding_release.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ReleaseTest(unittest.TestCase):
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):m.validate_records([{'id':'a','fields':{}},{'id':'a','fields':{}}])
    def test_private_path_rejected(self):
        with self.assertRaises(ValueError):m.validate_records([{'id':'a','fields':{'asset_path':'/data/250010098/private.glb'}}])
    def test_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);m.write_package(p,[{'id':'a','fields':{'asset_id':'a'}}],np.array([[1.,0.]],dtype=np.float32),{'model_id':'test'})
            self.assertEqual(m.verify_package(p)['count'],1)
            with (p/'vectors_000.npy').open('ab') as f:f.write(b'bad')
            with self.assertRaises(ValueError):m.verify_package(p)

if __name__=='__main__':unittest.main()
