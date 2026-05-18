"""
nltk_setup.py
=============
Called at app startup to ensure NLTK can find its data on Render.

Import this at the top of dashboard/app.py is not needed — it is
invoked automatically from src/features/preprocessor.py and any
module that uses NLTK.

On Render, NLTK_DATA is set to the nltk_data/ folder created by build.sh.
Locally, the default ~/nltk_data path is used.
"""

import os
import nltk

# Point NLTK to the project-local data folder when running on Render.
# On Render the working directory is the repo root.
_local_nltk = os.path.join(os.getcwd(), "nltk_data")
if os.path.isdir(_local_nltk) and _local_nltk not in nltk.data.path:
    nltk.data.path.insert(0, _local_nltk)

# Also respect the NLTK_DATA environment variable if set.
_env_nltk = os.environ.get("NLTK_DATA")
if _env_nltk and _env_nltk not in nltk.data.path:
    nltk.data.path.insert(0, _env_nltk)
