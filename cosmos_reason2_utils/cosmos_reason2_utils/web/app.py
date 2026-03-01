# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flask web application for Cosmos-Reason2 inference."""

import dataclasses
import logging
from pathlib import Path

import tyro
from flask import Flask

from cosmos_reason2_utils.web.routes import api

logger = logging.getLogger(__name__)


def create_app(
    vllm_url: str = "http://localhost:8000",
    workspace_cache: str = "/workspace_cache",
) -> Flask:
    """Create and configure the Flask application."""
    static_dir = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="/static")
    app.config["VLLM_URL"] = vllm_url
    app.config["WORKSPACE_CACHE"] = workspace_cache
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.register_blueprint(api)
    return app


@dataclasses.dataclass
class ServerArgs:
    """Cosmos-Reason2 Web UI server."""

    host: str = "0.0.0.0"
    """Server bind address."""
    port: int = 9900
    """Server port."""
    vllm_url: str = "http://localhost:8000"
    """vLLM server URL."""
    workspace_cache: str = "/workspace_cache"
    """Directory for persistent workspace cache."""
    debug: bool = False
    """Enable Flask debug mode."""


def main():
    args = tyro.cli(ServerArgs)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Cosmos-Reason2 Web UI on %s:%d", args.host, args.port)
    logger.info("vLLM backend: %s", args.vllm_url)
    app = create_app(vllm_url=args.vllm_url, workspace_cache=args.workspace_cache)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
