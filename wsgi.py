import sys
import os

# Add your project directory to the sys.path
project_home = u'/home/mhamzaa/MindGuide'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# Import your Flask app
from app import app as application

# Initialize cleanup thread
if __name__ == '__main__':
    from app import cleanup_temp_files
    import threading
    cleanup_thread = threading.Thread(target=cleanup_temp_files, daemon=True)
    cleanup_thread.start() 