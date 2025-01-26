"""This module provides methods for moving data between the local machine, the NAS drive (SMB protocol) and the Sun lab
BioHPC cluster (SFTP protocol).
"""

import hashlib
from pathlib import Path

from tqdm import tqdm
