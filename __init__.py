import os
import sys

# Add node root to sys.path for internal module resolution
NODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if NODE_ROOT not in sys.path:
    sys.path.insert(0, NODE_ROOT)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Define the web directory for JavaScript extensions
WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
