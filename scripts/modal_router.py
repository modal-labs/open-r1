# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from modal import App, Sandbox
from pydantic import BaseModel


class BatchRequest(BaseModel):
    """
    BatchRequest is a data model representing a batch processing request.

    Attributes:
        scripts (list[str]): A list of script names or paths to be executed.
        languages (list[str]): The programming languages for each script in the list.
        timeout (int): The maximum allowed execution time for each script in seconds.
        request_timeout (int): The maximum allowed time for the entire batch request in seconds.
    """

    scripts: list[str]
    languages: list[str]
    timeout: int
    request_timeout: int


class ScriptResult(BaseModel):
    """
    ScriptResult is a Pydantic model that represents the result of a script execution.
    Attributes:
        text (Optional[str]): The output text from the script execution.
        stdout (Optional[str]): The stdout from the script execution.
        exception_str (Optional[str]): An optional string that captures the exception
            message or details if an error occurred during the script's execution.
    """

    text: Optional[str]
    stdout: Optional[str]
    exception_str: Optional[str]


def create_app(args):
    """
    Creates and configures a FastAPI application instance.
    Args:
        args: An object containing configuration parameters for the application
    Returns:
        FastAPI: A configured FastAPI application instance
    """
    app = FastAPI()

    app.state.sandbox_semaphore = asyncio.Semaphore(args.max_num_sandboxes)

    modal_app = App.lookup("code-interpreter", create_if_missing=True)

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}

    def _run_script(
        sandbox: Sandbox, script: str, language: str, request_timeout: int
    ) -> str:
        if language == "python":
            tmp_file = f"/tmp/{uuid.uuid4()}.py"
            with sandbox.open(tmp_file, "w") as f:
                f.write(script)
                p = sandbox.exec("python", tmp_file, timeout=request_timeout)
                return p
        else:
            raise ValueError(f"Unsupported language: {language}")

    @app.post("/execute_batch")
    async def execute_batch(batch: BatchRequest, request: Request):
        semaphore = request.app.state.sandbox_semaphore
        timeout = batch.timeout
        request_timeout = batch.request_timeout
        asyncio_timeout = batch.timeout + 1

        async def run_script(script: str, language: str) -> ScriptResult:
            async with semaphore:
                sandbox = await asyncio.to_thread(
                    Sandbox.create,
                    app=modal_app,
                    timeout=timeout,
                    verbose=True,
                )
                execution = await asyncio.wait_for(
                    asyncio.to_thread(
                        _run_script,
                        sandbox,
                        script,
                        language,
                        request_timeout,
                    ),
                    timeout=asyncio_timeout,
                )
                sandbox.terminate()
                return ScriptResult(
                    text=str(execution.wait()),
                    stdout=execution.stdout.read(),
                    exception_str=execution.stderr.read(),
                )

        tasks = [
            run_script(script, lang)
            for script, lang in zip(batch.scripts, batch.languages)
        ]
        return await asyncio.gather(*tasks)

    return app


def parse_args():
    """
    Parse command-line arguments for the modal_router script.

    Arguments:
        --host (str): The hostname or IP address to bind the server to. Defaults to "0.0.0.0" (binds to all interfaces).
        --port (int): The port number on which the server will listen. Defaults to 8000.
        --max_num_sandboxes (int): The maximum number of sandboxes that can be created or managed simultaneously. Defaults to 20.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_num_sandboxes", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)

    uvicorn.run(app, host=args.host, port=args.port)
