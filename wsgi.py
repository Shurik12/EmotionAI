import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import create_app

app = create_app()

if __name__ == "__main__":
    app.run()
