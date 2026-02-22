"""Allow running the API with: python -m packages.api"""

import os
import uvicorn

from .dependencies import get_config


def main():
    """Run the API server."""
    config = get_config()
    uvicorn.run(
        "packages.api.main:app",
        host=config["api_host"],
        port=config["api_port"],
        reload=True,
    )


if __name__ == "__main__":
    main()
