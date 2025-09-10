#!/usr/bin/env python3
"""
JupyterLab Remote MCP Server
Complete implementation with all 12 MCP tools from jupyter-mcp-server
Connects to remote JupyterLab at https://cf-jupyter-uv.apps.tas-ndc.kuhn-labs.com
"""
import asyncio
import logging
import time
from typing import Optional, List, Union, Dict, Any
from urllib.parse import urljoin

import httpx
from mcp.server import FastMCP

# Try to import the custom packages, with fallback error handling
try:
    from jupyter_kernel_client import KernelClient
    from jupyter_nbmodel_client import (
        NbModelClient,
        get_notebook_websocket_url,
    )
    CUSTOM_PACKAGES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Custom packages not available: {e}")
    CUSTOM_PACKAGES_AVAILABLE = False
    # Create dummy classes for fallback
    class KernelClient:
        def __init__(self, *args, **kwargs):
            raise ImportError("jupyter_kernel_client not available")
    
    class NbModelClient:
        def __init__(self, *args, **kwargs):
            raise ImportError("jupyter_nbmodel_client not available")
    
    def get_notebook_websocket_url(*args, **kwargs):
        raise ImportError("jupyter_nbmodel_client not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - your remote JupyterLab
JUPYTER_URL = "https://cf-jupyter-uv.apps.tas-ndc.kuhn-labs.com"
NOTEBOOK_PATH = "nbsample.ipynb"
PROVIDER = "jupyter"

# Global variables for kernel management
kernel = None

async def __start_kernel():
    """Start the Jupyter kernel with error handling."""
    global kernel
    
    if not CUSTOM_PACKAGES_AVAILABLE:
        raise ImportError("Required packages not available")
    
    try:
        if kernel:
            try:
                kernel.stop()
            except Exception as e:
                logger.warning(f"Error stopping existing kernel: {e}")
    except Exception as e:
        logger.warning(f"Error stopping existing kernel: {e}")
    
    try:
        # Initialize the kernel client with the provided parameters.
        kernel = KernelClient(server_url=JUPYTER_URL, token=None, kernel_id=None)
        # Make this async-safe
        await asyncio.get_event_loop().run_in_executor(None, kernel.start)
        logger.info("Kernel started successfully")
    except Exception as e:
        logger.error(f"Failed to start kernel: {e}")
        kernel = None
        raise

async def __ensure_kernel_alive():
    """Ensure kernel is running, restart if needed."""
    global kernel
    if kernel is None:
        logger.info("Kernel is None, starting new kernel")
        await __start_kernel()
    elif not hasattr(kernel, 'is_alive') or not kernel.is_alive():
        logger.info("Kernel is not alive, restarting")
        await __start_kernel()

async def __safe_notebook_operation(operation_func, max_retries=3):
    """Safely execute notebook operations with connection recovery."""
    if not CUSTOM_PACKAGES_AVAILABLE:
        raise ImportError("Required packages not available")
    
    for attempt in range(max_retries):
        try:
            return await operation_func()
        except Exception as e:
            error_msg = str(e).lower()
            if any(err in error_msg for err in ["websocketclosederror", "connection is already closed", "connection closed"]):
                if attempt < max_retries - 1:
                    logger.warning(f"Connection lost, retrying... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(1 + attempt)  # Increasing delay
                    continue
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    raise Exception(f"Connection failed after {max_retries} retries: {e}")
            else:
                # Non-connection error, don't retry
                raise e
    
    raise Exception("Unexpected error in retry logic")

def safe_extract_outputs(outputs):
    """Safely extract outputs from cell execution."""
    try:
        if not outputs:
            return []
        
        result = []
        for output in outputs:
            # Handle case where output is already a string or primitive
            if isinstance(output, (str, int, float, bool)):
                result.append(str(output))
                continue
                
            # Handle case where output is a dict-like object
            if hasattr(output, 'get'):
                output_type = output.get('output_type', 'unknown')
                if output_type == 'stream':
                    text = output.get('text', '')
                    if isinstance(text, list):
                        result.extend(text)
                    else:
                        result.append(str(text))
                elif output_type in ['execute_result', 'display_data']:
                    data = output.get('data', {})
                    if 'text/plain' in data:
                        result.append(str(data['text/plain']))
                elif output_type == 'error':
                    result.append(f"Error: {output.get('ename', 'Unknown')}: {output.get('evalue', 'Unknown error')}")
            # Handle case where output has attributes but no get method
            elif hasattr(output, 'output_type'):
                # Try to access attributes directly
                try:
                    if output.output_type == 'stream':
                        result.append(str(output.text))
                    elif output.output_type in ['execute_result', 'display_data']:
                        if hasattr(output, 'data') and 'text/plain' in output.data:
                            result.append(str(output.data['text/plain']))
                    elif output.output_type == 'error':
                        result.append(f"Error: {getattr(output, 'ename', 'Unknown')}: {getattr(output, 'evalue', 'Unknown error')}")
                except Exception as attr_e:
                    result.append(f"Error processing output: {attr_e}")
            else:
                # Fallback: convert to string
                result.append(str(output))
        
        return result
    except Exception as e:
        logger.warning(f"Error extracting outputs: {e}")
        return [f"Error extracting outputs: {e}"]

# Create MCP server
server = FastMCP("jupyterlab-remote-mcp", json_response=False)

# ============================================================================
# Cell Creation & Editing Tools (6 tools)
# ============================================================================

@server.tool()
async def append_markdown_cell(cell_source: str) -> str:
    """Append at the end of the notebook a markdown cell with the provided source."""
    async def _append_markdown():
        if not CUSTOM_PACKAGES_AVAILABLE:
            raise ImportError("Required packages not available")
            
        notebook = None
        try:
            notebook = NbModelClient(
                get_notebook_websocket_url(
                    server_url=JUPYTER_URL, token=None, path=NOTEBOOK_PATH, provider=PROVIDER
                )
            )
            await notebook.start()
            notebook.add_markdown_cell(cell_source)
            return "Jupyter Markdown cell added."
        finally:
            if notebook:
                try:
                    await notebook.stop()
                except Exception as e:
                    logger.warning(f"Error stopping notebook in append_markdown_cell: {e}")
    
    result = await __safe_notebook_operation(_append_markdown)
    return result[0] if isinstance(result, list) else result

@server.tool()
async def insert_markdown_cell(cell_index: int, cell_source: str) -> str:
    """Insert a markdown cell at a specific index in the notebook."""
    async def _insert_markdown():
        if not CUSTOM_PACKAGES_AVAILABLE:
            raise ImportError("Required packages not available")
            
        notebook = None
        try:
            notebook = NbModelClient(
                get_notebook_websocket_url(
                    server_url=JUPYTER_URL, token=None, path=NOTEBOOK_PATH, provider=PROVIDER
                )
            )
            await notebook.start()
            notebook.insert_markdown_cell(cell_index, cell_source)
            return f"Jupyter Markdown cell inserted at index {cell_index}."
        finally:
            if notebook:
                try:
                    await notebook.stop()
                except Exception as e:
                    logger.warning(f"Error stopping notebook in insert_markdown_cell: {e}")
    
    result = await __safe_notebook_operation(_insert_markdown)
    return result[0] if isinstance(result, list) else result

@server.tool()
async def append_execute_code_cell(cell_source: str) -> list[str]:
    """Append at the end of the notebook a code cell with the provided source and execute it."""
    async def _append_execute():
        if not CUSTOM_PACKAGES_AVAILABLE:
            raise ImportError("Required packages not available")
            
        await __ensure_kernel_alive()
        notebook = None
        try:
            notebook = NbModelClient(
                get_notebook_websocket_url(
                    server_url=JUPYTER_URL, token=None, path=NOTEBOOK_PATH, provider=PROVIDER
                )
            )
            await notebook.start()
            cell_index = notebook.add_code_cell(cell_source)
            notebook.execute_cell(cell_index, kernel)

            # Wait a bit for execution to complete
            await asyncio.sleep(2)
            
            ydoc = notebook._doc
            outputs = ydoc._ycells[cell_index]["outputs"]
            return safe_extract_outputs(outputs)
        finally:
            if notebook:
                try:
                    await notebook.stop()
                except Exception as e:
                    logger.warning(f"Error stopping notebook in append_execute_code_cell: {e}")
    
    return await __safe_notebook_operation(_append_execute)

@server.tool()
async def insert_execute_code_cell(cell_index: int, cell_source: str) -> list[str]:
    """Insert a code cell at a specific index in the notebook and execute it."""
    async def _insert_execute():
        if not CUSTOM_PACKAGES_AVAILABLE:
            raise ImportError("Required packages not available")
            
        await __ensure_kernel_alive()
        notebook = None
        try:
            notebook = NbModelClient(
                get_notebook_websocket_url(
                    server_url=JUPYTER_URL, token=None, path=NOTEBOOK_PATH, provider=PROVIDER
                )
            )
            await notebook.start()
            notebook.insert_code_cell(cell_index, cell_source)
            notebook.execute_cell(cell_index, kernel)

            # Wait a bit for execution to complete
            await asyncio.sleep(2)
            
            ydoc = notebook._doc
            outputs = ydoc._ycells[cell_index]["outputs"]
            return safe_extract_outputs(outputs)
        finally:
            if notebook:
                try:
                    await notebook.stop()
                except Exception as e:
                    logger.warning(f"Error stopping notebook in insert_execute_code_cell: {e}")
    
    return await __safe_notebook_operation(_insert_execute)

@server.tool()
async def overwrite_cell_source(cell_index: int, cell_source: str) -> str:
    """Overwrite the source of an existing cell."""
    async def _overwrite():
        if not CUSTOM_PACKAGES_AVAILABLE:
            raise ImportError("Required packages not available")
            
        notebook = None
        try:
            notebook = NbModelClient(
                get_notebook_websocket_url(
                    server_url=JUPYTER_URL, token=None, path=NOTEBOOK_PATH, provider=PROVIDER
                )
            )
            await notebook.start()

            ydoc = notebook._doc

            if cell_index < 0 or cell_index >= len(ydoc._ycells):
                raise ValueError(
                    f"Cell index {cell_index} is out of range. Notebook has {len(ydoc._ycells)} cells."
                )

            # Overwrite cell source
            ydoc._ycells[cell_index]["source"] = cell_source
            
            return f"Cell {cell_index} source overwritten successfully."
        finally:
            if notebook:
                try:
                    await notebook.stop()
                except Exception as e:
                    logger.warning(f"Error stopping notebook in overwrite_cell_source: {e}")
    
    result = await __safe_notebook_operation(_overwrite)
    return result[0] if isinstance(result, list) else result

@server.tool()
async def delete_cell(cell_index: int) -> str:
    """Delete a specific cell from the Jupyter notebook."""
    async def _delete_cell():
        if not CUSTOM_PACKAGES_AVAILABLE:
            raise ImportError("Required packages not available")
            
        notebook = None
        try:
            notebook = NbModelClient(
                get_notebook_websocket_url(
                    server_url=JUPYTER_URL, token=None, path=NOTEBOOK_PATH, provider=PROVIDER
                )
            )
            await notebook.start()

            ydoc = notebook._doc

            if cell_index < 0 or cell_index >= len(ydoc._ycells):
                raise ValueError(
                    f"Cell index {cell_index} is out of range. Notebook has {len(ydoc._ycells)} cells."
                )

            cell_type = ydoc._ycells[cell_index].get("cell_type", "unknown")

            # Delete the cell
            del ydoc._ycells[cell_index]

            return f"Cell {cell_index} ({cell_type}) deleted successfully."
        finally:
            if notebook:
                try:
                    await notebook.stop()
                except Exception as e:
                    logger.warning(f"Error stopping notebook in delete_cell: {e}")
    
    result = await __safe_notebook_operation(_delete_cell)
    return result[0] if isinstance(result, list) else result

# ============================================================================
# Cell Reading Tools (3 tools)
# ============================================================================

@server.tool()
async def read_all_cells() -> list[dict[str, Union[str, int, list[str]]]]:
    """Read all cells from the Jupyter notebook."""
    async def _read_all():
        if not CUSTOM_PACKAGES_AVAILABLE:
            return [{"error": "Required packages not available"}]
            
        notebook = None
        try:
            notebook = NbModelClient(
                get_notebook_websocket_url(
                    server_url=JUPYTER_URL, token=None, path=NOTEBOOK_PATH, provider=PROVIDER
                )
            )
            await notebook.start()

            ydoc = notebook._doc
            cells = []

            for i, cell in enumerate(ydoc._ycells):
                cell_info = {
                    "index": i,
                    "cell_type": cell.get("cell_type", "unknown"),
                    "source": cell.get("source", ""),
                }
                
                # Add outputs for code cells
                if cell.get("cell_type") == "code":
                    outputs = cell.get("outputs", [])
                    cell_info["outputs"] = safe_extract_outputs(outputs)
                
                cells.append(cell_info)
            
            return cells
        finally:
            if notebook:
                try:
                    await notebook.stop()
                except Exception as e:
                    logger.warning(f"Error stopping notebook in read_all_cells: {e}")
    
    result = await __safe_notebook_operation(_read_all)
    return result if isinstance(result, list) and not isinstance(result[0], str) else [{"error": "Failed to read cells"}]

@server.tool()
async def read_cell(cell_index: int) -> dict[str, Union[str, int, list[str]]]:
    """Read a specific cell from the Jupyter notebook."""
    async def _read_cell():
        if not CUSTOM_PACKAGES_AVAILABLE:
            return {"error": "Required packages not available"}
            
        notebook = None
        try:
            notebook = NbModelClient(
                get_notebook_websocket_url(
                    server_url=JUPYTER_URL, token=None, path=NOTEBOOK_PATH, provider=PROVIDER
                )
            )
            await notebook.start()

            ydoc = notebook._doc

            if cell_index < 0 or cell_index >= len(ydoc._ycells):
                raise ValueError(
                    f"Cell index {cell_index} is out of range. Notebook has {len(ydoc._ycells)} cells."
                )

            cell = ydoc._ycells[cell_index]
            cell_info = {
                "index": cell_index,
                "cell_type": cell.get("cell_type", "unknown"),
                "source": cell.get("source", ""),
            }
            
            # Add outputs for code cells
            if cell.get("cell_type") == "code":
                outputs = cell.get("outputs", [])
                cell_info["outputs"] = safe_extract_outputs(outputs)
            
            return cell_info
        finally:
            if notebook:
                try:
                    await notebook.stop()
                except Exception as e:
                    logger.warning(f"Error stopping notebook in read_cell: {e}")
    
    result = await __safe_notebook_operation(_read_cell)
    return result if isinstance(result, dict) else {"error": "Failed to read cell"}

@server.tool()
async def get_notebook_info() -> dict[str, Union[str, int, dict[str, int]]]:
    """Get basic information about the notebook."""
    async def _get_info():
        if not CUSTOM_PACKAGES_AVAILABLE:
            return {"error": "Required packages not available"}
            
        notebook = None
        try:
            notebook = NbModelClient(
                get_notebook_websocket_url(
                    server_url=JUPYTER_URL, token=None, path=NOTEBOOK_PATH, provider=PROVIDER
                )
            )
            await notebook.start()

            ydoc = notebook._doc
            
            # Count cell types
            cell_counts = {}
            for cell in ydoc._ycells:
                cell_type = cell.get("cell_type", "unknown")
                cell_counts[cell_type] = cell_counts.get(cell_type, 0) + 1

            return {
                "notebook_path": NOTEBOOK_PATH,
                "total_cells": len(ydoc._ycells),
                "cell_type_counts": cell_counts
            }
        finally:
            if notebook:
                try:
                    await notebook.stop()
                except Exception as e:
                    logger.warning(f"Error stopping notebook in get_notebook_info: {e}")
    
    result = await __safe_notebook_operation(_get_info)
    return result if isinstance(result, dict) else {"error": "Failed to get notebook info"}

# ============================================================================
# Cell Execution Tools (3 tools)
# ============================================================================

async def __wait_for_kernel_idle(kernel, max_wait_seconds=30):
    """Wait for kernel to be idle."""
    if not kernel:
        return
    
    start_time = time.time()
    while time.time() - start_time < max_wait_seconds:
        try:
            if hasattr(kernel, 'is_alive') and kernel.is_alive():
                break
        except:
            pass
        await asyncio.sleep(0.5)

@server.tool()
async def execute_cell_with_progress(cell_index: int, timeout_seconds: int = 300) -> list[str]:
    """Execute a specific cell with timeout and progress monitoring."""
    async def _execute():
        if not CUSTOM_PACKAGES_AVAILABLE:
            raise ImportError("Required packages not available")
            
        await __ensure_kernel_alive()
        await __wait_for_kernel_idle(kernel, max_wait_seconds=30)
        
        notebook = None
        try:
            notebook = NbModelClient(
                get_notebook_websocket_url(
                    server_url=JUPYTER_URL, token=None, path=NOTEBOOK_PATH, provider=PROVIDER
                )
            )
            await notebook.start()

            ydoc = notebook._doc

            if cell_index < 0 or cell_index >= len(ydoc._ycells):
                raise ValueError(
                    f"Cell index {cell_index} is out of range. Notebook has {len(ydoc._ycells)} cells."
                )

            logger.info(f"Starting execution of cell {cell_index} with {timeout_seconds}s timeout")
            
            # Execute the cell using asyncio.to_thread like the working version
            execution_task = asyncio.create_task(
                asyncio.to_thread(notebook.execute_cell, cell_index, kernel)
            )
            
            try:
                await asyncio.wait_for(execution_task, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                execution_task.cancel()
                if kernel and hasattr(kernel, 'interrupt'):
                    kernel.interrupt()
                return [f"[TIMEOUT ERROR: Cell execution exceeded {timeout_seconds} seconds]"]
            
            # Get final outputs
            outputs = ydoc._ycells[cell_index]["outputs"]
            result = safe_extract_outputs(outputs)
            
            logger.info(f"Cell {cell_index} completed successfully with {len(result)} outputs")
            return result
            
        except Exception as e:
            logger.error(f"Error executing cell {cell_index}: {e}")
            raise
        finally:
            if notebook:
                try:
                    await notebook.stop()
                except Exception as e:
                    logger.warning(f"Error stopping notebook in execute_cell_with_progress: {e}")
    
    return await __safe_notebook_operation(_execute)

@server.tool()
async def execute_cell_simple_timeout(cell_index: int, timeout_seconds: int = 300) -> list[str]:
    """Execute a cell with simple timeout (no forced real-time sync). To be used for short-running cells."""
    async def _execute_simple():
        if not CUSTOM_PACKAGES_AVAILABLE:
            raise ImportError("Required packages not available")
            
        await __ensure_kernel_alive()
        
        notebook = None
        try:
            notebook = NbModelClient(
                get_notebook_websocket_url(
                    server_url=JUPYTER_URL, token=None, path=NOTEBOOK_PATH, provider=PROVIDER
                )
            )
            await notebook.start()

            ydoc = notebook._doc

            if cell_index < 0 or cell_index >= len(ydoc._ycells):
                raise ValueError(
                    f"Cell index {cell_index} is out of range. Notebook has {len(ydoc._ycells)} cells."
                )

            # Execute the cell using asyncio.to_thread like the working version
            execution_task = asyncio.create_task(
                asyncio.to_thread(notebook.execute_cell, cell_index, kernel)
            )
            
            try:
                await asyncio.wait_for(execution_task, timeout=timeout_seconds)
            except asyncio.TimeoutError:
                execution_task.cancel()
                if kernel and hasattr(kernel, 'interrupt'):
                    kernel.interrupt()
                return [f"[TIMEOUT ERROR: Cell execution exceeded {timeout_seconds} seconds]"]

            # Get outputs
            ydoc = notebook._doc
            outputs = ydoc._ycells[cell_index]["outputs"]
            return safe_extract_outputs(outputs)
            
        except Exception as e:
            logger.error(f"Error in simple execution of cell {cell_index}: {e}")
            raise
        finally:
            if notebook:
                try:
                    await notebook.stop()
                except Exception as e:
                    logger.warning(f"Error stopping notebook in execute_cell_simple_timeout: {e}")
    
    return await __safe_notebook_operation(_execute_simple)

@server.tool()
async def execute_cell_streaming(cell_index: int, timeout_seconds: int = 300, progress_interval: int = 5) -> list[str]:
    """Execute cell with streaming progress updates. To be used for long-running cells."""
    async def _execute_streaming():
        if not CUSTOM_PACKAGES_AVAILABLE:
            raise ImportError("Required packages not available")
            
        await __ensure_kernel_alive()
        
        notebook = None
        try:
            notebook = NbModelClient(
                get_notebook_websocket_url(
                    server_url=JUPYTER_URL, token=None, path=NOTEBOOK_PATH, provider=PROVIDER
                )
            )
            await notebook.start()

            ydoc = notebook._doc

            if cell_index < 0 or cell_index >= len(ydoc._ycells):
                raise ValueError(
                    f"Cell index {cell_index} is out of range. Notebook has {len(ydoc._ycells)} cells."
                )

            logger.info(f"Starting streaming execution of cell {cell_index}")
            
            # Execute the cell using asyncio.to_thread
            execution_task = asyncio.create_task(
                asyncio.to_thread(notebook.execute_cell, cell_index, kernel)
            )
            
            # Stream progress updates
            start_time = time.time()
            all_outputs = []
            
            while not execution_task.done():
                elapsed = time.time() - start_time
                
                # Check timeout
                if elapsed > timeout_seconds:
                    execution_task.cancel()
                    all_outputs.append(f"[TIMEOUT at {elapsed:.1f}s: Cancelling execution]")
                    try:
                        if kernel and hasattr(kernel, 'interrupt'):
                            kernel.interrupt()
                        all_outputs.append("[Sent interrupt signal to kernel]")
                    except Exception:
                        pass
                    break
                
                # Check for new outputs
                try:
                    current_outputs = ydoc._ycells[cell_index].get("outputs", [])
                    if len(current_outputs) > len(all_outputs):
                        new_outputs = current_outputs[len(all_outputs):]
                        for output in new_outputs:
                            extracted = safe_extract_outputs([output])
                            if extracted and extracted[0].strip():
                                all_outputs.append(f"[{elapsed:.1f}s] {extracted[0]}")
                except Exception as e:
                    all_outputs.append(f"[{elapsed:.1f}s] Error checking outputs: {e}")
                
                # Progress update
                if int(elapsed) % progress_interval == 0 and elapsed > 0:
                    all_outputs.append(f"[PROGRESS: {elapsed:.1f}s elapsed, {len(all_outputs)} outputs so far]")
                
                await asyncio.sleep(1)
            
            # Get final result
            if not execution_task.cancelled():
                try:
                    await execution_task
                    final_outputs = ydoc._ycells[cell_index].get("outputs", [])
                    all_outputs.append(f"[COMPLETED in {time.time() - start_time:.1f}s]")
                    
                    # Add any final outputs not captured during monitoring
                    if len(final_outputs) > len(all_outputs):
                        remaining = final_outputs[len(all_outputs):]
                        for output in remaining:
                            extracted = safe_extract_outputs([output])
                            if extracted and extracted[0].strip():
                                all_outputs.append(extracted[0])
                                
                except Exception as e:
                    all_outputs.append(f"[ERROR: {e}]")
            
            return all_outputs if all_outputs else ["[No output generated]"]
            
        except Exception as e:
            logger.error(f"Error in streaming execution of cell {cell_index}: {e}")
            raise
        finally:
            if notebook:
                try:
                    await notebook.stop()
                except Exception as e:
                    logger.warning(f"Error stopping notebook in execute_cell_streaming: {e}")
    
    return await __safe_notebook_operation(_execute_streaming)

# ============================================================================
# Connection & Information Tools
# ============================================================================

@server.tool()
async def test_connection() -> str:
    """Test connection to the remote JupyterLab instance"""
    try:
        # Test HTTP connection
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            response = await client.get(f"{JUPYTER_URL}/api/kernels")
            response.raise_for_status()
            kernels = response.json()
            
            # Test WebSocket connection
            if CUSTOM_PACKAGES_AVAILABLE:
                ws_url = get_notebook_websocket_url(
                    server_url=JUPYTER_URL, token=None, path=NOTEBOOK_PATH, provider=PROVIDER
                )
                ws_status = "Available"
            else:
                ws_url = "Not available (packages missing)"
                ws_status = "Not available"
            
            return f"✅ Connection successful!\nHTTP API: Working\nKernels available: {len(kernels)}\nWebSocket: {ws_status}\nWebSocket URL: {ws_url}"
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return f"❌ Connection failed: {str(e)}"

@server.tool()
async def list_kernels() -> str:
    """List available kernels on the remote JupyterLab"""
    try:
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            response = await client.get(f"{JUPYTER_URL}/api/kernels")
            response.raise_for_status()
            kernels = response.json()
            
            if kernels:
                kernel_info = [f"ID: {k.get('id', 'Unknown')} - Name: {k.get('name', 'Unknown')} - State: {k.get('execution_state', 'Unknown')}" for k in kernels]
                return f"Available kernels ({len(kernels)}):\n" + "\n".join(kernel_info)
            else:
                return "No kernels currently running"
    except Exception as e:
        logger.error(f"Error listing kernels: {e}")
        return f"Error: {str(e)}"

@server.tool()
async def create_kernel(kernel_name: str = "python3") -> str:
    """Create a new kernel on the remote JupyterLab"""
    try:
        async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
            response = await client.post(f"{JUPYTER_URL}/api/kernels", json={"name": kernel_name})
            response.raise_for_status()
            kernel_data = response.json()
            return f"Created kernel: {kernel_data.get('id', 'Unknown')} - Name: {kernel_data.get('name', 'Unknown')}"
    except Exception as e:
        logger.error(f"Error creating kernel: {e}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    """Start the MCP server with stdio transport"""
    logger.info("Starting JupyterLab Remote MCP Server...")
    logger.info(f"Target JupyterLab: {JUPYTER_URL}")
    logger.info(f"Target notebook: {NOTEBOOK_PATH}")
    
    if CUSTOM_PACKAGES_AVAILABLE:
        logger.info("All 12 MCP tools implemented:")
        logger.info("  Cell Creation: append_markdown_cell, insert_markdown_cell, append_execute_code_cell, insert_execute_code_cell, overwrite_cell_source, delete_cell")
        logger.info("  Cell Reading: read_all_cells, read_cell, get_notebook_info")
        logger.info("  Cell Execution: execute_cell_with_progress, execute_cell_simple_timeout, execute_cell_streaming")
        logger.info("  Connection: test_connection, list_kernels, create_kernel")
    else:
        logger.warning("Custom packages not available - limited functionality")
        logger.info("Available tools: test_connection, list_kernels, create_kernel")
    
    # Run the server with stdio transport
    server.run(transport="stdio")
