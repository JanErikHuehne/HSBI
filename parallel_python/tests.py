import unittest 
import subprocess
from pathlib import Path
import os
DATA_DIR = '/home/ge84yes/master_thesis/HSBI/data/hsbi_data/raw_results'
class Tests(unittest.TestCase):
    
    def test_simulations(self):
        script = Path('/home/ge84yes/master_thesis/HSBI/test_data/sim2.sh').resolve()
        print(script.exists())
        self.assertTrue(Path(DATA_DIR).exists())
        subprocess.run([script])
        results_generated = [DATA_DIR + "/" + name for name in os.listdir(DATA_DIR) if os.path.isfile(DATA_DIR + "/" + name)]
        self.assertEqual(len(results_generated), 30)
        for result in results_generated:
            os.remove(result)
if __name__ == '__main__':
    unittest.main()