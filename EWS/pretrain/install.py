import os
import subprocess
import sys
index_url = os.environ.get('TORCH_INDEX_URL', 'https://download.pytorch.org/whl/nightly/cu128')
extra = ['--pre'] if os.environ.get('TORCH_PIP_PRE', '1') not in ('0', 'false', 'False') else []
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall', *extra, 'torch', '--index-url', index_url])
