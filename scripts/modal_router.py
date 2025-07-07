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
        exception_str (Optional[str]): An optional string that captures the exception
            message or details if an error occurred during the script's execution.
    """

    text: Optional[str]
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

    def run_script(
        sandbox: Sandbox, script: str, language: str, request_timeout: int
    ) -> str:
        if language == "python":
            return sandbox.exec(
                "python", "-c", script, timeout=request_timeout
            ).stdout.read()
        elif language == "javascript":
            return sandbox.exec(
                "node", "-e", script, timeout=request_timeout
            ).stdout.read()
        elif language == "r":
            return sandbox.exec(
                "R", "-e", script, timeout=request_timeout
            ).stdout.read()
        elif language == "java":
            # For Java, we need to create a temporary file and compile/run it
            temp_file = f"/tmp/temp_{hash(script) % 1000000}.java"
            sandbox.exec(
                "sh", "-c", f'echo "{script}" > {temp_file}', timeout=request_timeout
            )
            class_name = f"Temp{hash(script) % 1000000}"
            sandbox.exec("javac", temp_file, timeout=request_timeout)
            return sandbox.exec(
                "java", "-cp", "/tmp", class_name, timeout=request_timeout
            ).stdout.read()
        elif language == "bash":
            return sandbox.exec(
                "bash", "-c", script, timeout=request_timeout
            ).stdout.read()
        elif language == "cpp":
            # For C++, we need to create a temporary file and compile/run it
            temp_file = f"/tmp/temp_{hash(script) % 1000000}.cpp"
            sandbox.exec(
                "sh", "-c", f'echo "{script}" > {temp_file}', timeout=request_timeout
            )
            sandbox.exec(
                "g++",
                "-o",
                f"/tmp/temp_{hash(script) % 1000000}",
                temp_file,
                timeout=request_timeout,
            )
            return sandbox.exec(
                f"/tmp/temp_{hash(script) % 1000000}", timeout=request_timeout
            ).stdout.read()
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
                try:
                    sandbox = await asyncio.to_thread(
                        Sandbox.create, app=modal_app, timeout=timeout
                    )
                    execution = await asyncio.wait_for(
                        asyncio.to_thread(
                            run_script,
                            sandbox,
                            script,
                            language,
                            request_timeout,
                            timeout=asyncio_timeout,
                        )
                    )
                    return ScriptResult(
                        text=execution.stdout.read(), exception_str=None
                    )

                except Exception as e:
                    return ScriptResult(text=None, exception_str=str(e))

                finally:
                    try:
                        await sandbox.terminate()
                    except Exception:
                        pass

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
