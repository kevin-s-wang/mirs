from jsonargparse import CLI
from typing import Optional

def main(
    path: str,
):
    """Upload images to MIRS.

    Args:
        path: Directory or path to a zip file.
    """
    print(f'Path: {path}')

if __name__ == '__main__':
    CLI(main)