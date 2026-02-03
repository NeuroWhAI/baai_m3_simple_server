from collections import defaultdict
from cachetools import LRUCache
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List, Dict, Any
import asyncio
from fastapi import FastAPI, Request, HTTPException, Depends
from starlette.status import HTTP_504_GATEWAY_TIMEOUT, HTTP_429_TOO_MANY_REQUESTS
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time
import logging
import os
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import torch
from contextlib import asynccontextmanager
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("embedding-service")


# Configuration
class Config:
    # Model settings
    MODEL_DIR = os.environ.get("MODEL_DIR", "models")
    MODEL_NAME = os.environ.get("MODEL_NAME", "bge-m3-onnx")
    ONNX_FILE = os.environ.get("ONNX_FILE", "model.onnx")
    DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # Processing settings
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))  # GPU batch size based on VRAM
    MAX_LENGTH = int(
        os.environ.get("MAX_LENGTH", "5000")
    )  # Max context length for embeddings

    # Queue and timeout settings
    MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "100"))
    MAX_REQUEST = int(os.environ.get("MAX_REQUEST", "30"))  # Max pending requests
    REQUEST_FLUSH_TIMEOUT = float(
        os.environ.get("REQUEST_FLUSH_TIMEOUT", "0.05")
    )  # Seconds
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30"))  # Seconds
    GPU_TIMEOUT = int(os.environ.get("GPU_TIMEOUT", "60"))  # Seconds

    # Server settings
    HOST = os.environ.get("HOST", "localhost")
    PORT = int(os.environ.get("PORT", "3000"))
    WORKERS = int(os.environ.get("WORKERS", "1"))  # Number of worker processes
    ENABLE_CORS = os.environ.get("ENABLE_CORS", "False").lower() in ("true", "1", "yes")

    # Worker threads for the ThreadPoolExecutor
    WORKER_THREADS = int(os.environ.get("WORKER_THREADS", "4"))

    # Cache settings
    CACHE_MAX_SIZE = int(os.environ.get("CACHE_MAX_SIZE", "10000"))
    CACHE_MAX_TEXT_LENGTH = int(os.environ.get("CACHE_MAX_TEXT_LENGTH", "1000"))
    CACHE_HIT_COUNTER_SIZE = int(os.environ.get("CACHE_HIT_COUNTER_SIZE", "10"))
    CACHE_HIT_COUNTER_MAX = int(os.environ.get("CACHE_HIT_COUNTER_MAX", "10"))


class M3ModelWrapper:
    """Wrapper for the BGEM3FlagModel to handle embedding operations."""

    def __init__(self, model_dir: str, onnx_file: str, device: str = "cuda"):
        logger.info(f"Initializing model {model_dir} on {device})")
        try:
            self.device = device

            providers = ["CPUExecutionProvider"]
            so = ort.SessionOptions()
            if device == "cuda":
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    providers = [
                        (
                            "CUDAExecutionProvider",
                            {
                                "device_id": 0,
                                "arena_extend_strategy": "kSameAsRequested",
                                "cudnn_conv_algo_search": "EXHAUSTIVE",
                                "do_copy_in_default_stream": True,
                            },
                        ),
                        "CPUExecutionProvider",
                    ]

                    so.enable_mem_pattern = True
                    so.enable_mem_reuse = True
                    so.add_session_config_entry(
                        "memory.enable_memory_arena_shrinkage", "gpu:0"
                    )
                    so.add_session_config_entry(
                        "session.use_device_allocator_for_initializers", "1"
                    )
                    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    so.graph_optimization_level = (
                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    )
                else:
                    logger.warning(
                        "CUDAExecutionProvider not available. Using CPUExecutionProvider only."
                    )
                    self.device = "cpu"

            self.ort_session = ort.InferenceSession(
                os.path.join(model_dir, onnx_file), providers=providers, sess_options=so
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            logger.info("Model initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def warm_up(self, sequence_length=64, num_samples=2):
        """Warm up the model with dummy requests to initialize cuda kernels."""
        logger.info(f"Warming up model with {num_samples} dummy samples...")
        try:
            dummy_texts = [
                " ".join(["warm"] * (sequence_length // 5)) for _ in range(num_samples)
            ]

            start_time = time.time()
            _ = self.embed(dummy_texts)

            warm_up_time = time.time() - start_time
            logger.info(f"Model warm-up completed in {warm_up_time:.2f}s")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def embed(self, sentences: List[str]) -> Dict[str, List]:
        """Generate both dense and sparse embeddings for a list of sentences."""
        try:
            start_time = time.time()

            dense_vecs = []
            lexical_weights = []

            for i in range(0, len(sentences), Config.BATCH_SIZE):
                batch = sentences[i : i + Config.BATCH_SIZE]

                inputs = self.tokenizer(
                    batch,
                    padding="longest",
                    return_tensors="np",
                    truncation=True,
                    max_length=Config.MAX_LENGTH,
                )
                inputs_onnx = {
                    k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in inputs.items()
                }

                outputs = self.ort_session.run(None, inputs_onnx)

                dense_vecs.extend(outputs[0].tolist())

                token_weights = outputs[1].squeeze(-1)
                lexical_weights.extend(
                    map(
                        self.__process_token_weights,
                        token_weights,
                        inputs["input_ids"].tolist(),
                    )
                )

            processing_time = time.time() - start_time
            logger.debug(
                f"Embedding {len(sentences)} sentences took {processing_time:.2f}s"
            )

            return {
                "dense_vecs": dense_vecs,
                "lexical_weights": lexical_weights,
                "processing_time": processing_time,
            }
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise

    def __process_token_weights(self, token_weights: np.ndarray, input_ids: list):
        # conver to dict
        result = defaultdict(int)
        unused_tokens = set(
            [
                self.tokenizer.cls_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
            ]
        )
        for w, idx in zip(token_weights, input_ids):
            if idx not in unused_tokens and w > 0:
                idx = str(idx)
                # w = int(w)
                if w > result[idx]:
                    result[idx] = w
        return result


# --- Pydantic Models ---
class EmbedRequest(BaseModel):
    sentences: List[str] = Field(
        ..., min_items=1, description="List of sentences to embed"
    )


class EmbedResponse(BaseModel):
    dense_vecs: List[List[float]]
    lexical_weights: List[Dict[str, float]]
    processing_time: float


class HealthResponse(BaseModel):
    status: str
    queue_size: int
    active_requests: int
    total_requests: int
    error_count: int
    uptime: float


class RequestProcessor:
    """Handles queueing and processing of embedding requests."""

    def __init__(self, model: M3ModelWrapper):
        self.model = model
        self.queue = asyncio.Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.response_futures: Dict[str, asyncio.Future] = {}
        self.processing_loop_task = None
        self.processing_loop_started = False
        self.executor = ThreadPoolExecutor(max_workers=Config.WORKER_THREADS)
        self.gpu_lock = asyncio.Semaphore(1)  # Semaphore for GPU sync usage
        self.start_time = time.time()
        self.active_requests = 0
        self.request_counter = 0
        self.error_counter = 0
        self.embedding_cache = (
            LRUCache(maxsize=Config.CACHE_MAX_SIZE)
            if Config.CACHE_MAX_SIZE > 0
            else None
        )
        self.hit_counter_cache = (
            LRUCache(maxsize=Config.CACHE_HIT_COUNTER_SIZE)
            if Config.CACHE_HIT_COUNTER_SIZE > 0
            else None
        )
        self.cache_lock = threading.Lock()

    def _get_cached_embedding(self, sentence: str):
        if not self.embedding_cache:
            return None
        with self.cache_lock:
            return self.embedding_cache.get(sentence)

    def _set_cached_embedding(self, sentence: str, value):
        if not self.embedding_cache:
            return
        if len(sentence) > Config.CACHE_MAX_TEXT_LENGTH:
            return
        with self.cache_lock:
            self.embedding_cache[sentence] = value

    def _record_hit(self, sentence: str) -> int:
        if not self.hit_counter_cache:
            return 0
        with self.cache_lock:
            count = self.hit_counter_cache.get(sentence, 0)
            count = min(count + 1, Config.CACHE_HIT_COUNTER_MAX)
            self.hit_counter_cache[sentence] = count
            return count

    async def ensure_processing_loop_started(self):
        """Ensures the request processing loop is running."""
        if not self.processing_loop_started:
            logger.info("Starting processing loop")
            self.processing_loop_task = asyncio.create_task(self.processing_loop())
            self.processing_loop_started = True

    async def processing_loop(self):
        """Main processing loop that handles batching of requests."""
        while True:
            try:
                requests, request_ids = [], []
                start_time = asyncio.get_event_loop().time()
                timeout = max(0.001, Config.REQUEST_FLUSH_TIMEOUT)

                # Collect requests until batch is full or timeout occurs
                while len(requests) < Config.MAX_REQUEST:
                    try:
                        req_data, req_id = await asyncio.wait_for(
                            self.queue.get(), timeout=timeout
                        )
                        requests.append(req_data)
                        request_ids.append(req_id)
                    except asyncio.TimeoutError:
                        break

                    timeout = Config.REQUEST_FLUSH_TIMEOUT - (
                        asyncio.get_event_loop().time() - start_time
                    )
                    if timeout <= 0:
                        break


                if requests:
                    await self.process_requests(requests, request_ids)
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(0.1)  # Prevent tight loop in case of errors

    async def process_requests(self, requests, request_ids):
        """Process batched embedding requests."""
        # Combine all sentences into a single batch
        all_sentences = []
        for req in requests:
            all_sentences.extend(req.sentences)
        request_sizes = [len(req.sentences) for req in requests]

        try:
            cached_results = [None] * len(all_sentences)
            missing_sentences = []
            missing_positions = []
            missing_map = {}

            for pos, sentence in enumerate(all_sentences):
                hit_count = self._record_hit(sentence)
                cached = self._get_cached_embedding(sentence)
                if cached:
                    cached_results[pos] = cached
                    continue

                missing_index = missing_map.get(sentence)
                if missing_index is None:
                    missing_map[sentence] = len(missing_sentences)
                    missing_sentences.append((sentence, hit_count))
                    missing_positions.append([pos])
                else:
                    missing_positions[missing_index].append(pos)

            processing_time = 0.0
            if missing_sentences:
                sentences_to_embed = [sentence for sentence, _ in missing_sentences]
                result = await self.run_with_semaphore(
                    self.model.embed, sentences_to_embed
                )
                processing_time = result["processing_time"]
                for miss_idx, (sentence, hit_count) in enumerate(missing_sentences):
                    dense_vec = result["dense_vecs"][miss_idx]
                    lexical_weight = result["lexical_weights"][miss_idx]
                    if hit_count >= 2:
                        self._set_cached_embedding(
                            sentence, (dense_vec, lexical_weight)
                        )
                    for pos in missing_positions[miss_idx]:
                        cached_results[pos] = (dense_vec, lexical_weight)

            start_idx = 0
            for i, size in enumerate(request_sizes):
                if i < len(request_ids):
                    end_idx = start_idx + size
                    dense_vecs = [
                        cached_results[j][0] for j in range(start_idx, end_idx)
                    ]
                    lexical_weights = [
                        cached_results[j][1] for j in range(start_idx, end_idx)
                    ]
                    partial_result = {
                        "dense_vecs": dense_vecs,
                        "lexical_weights": lexical_weights,
                        "processing_time": processing_time,
                    }
                    self.response_futures[request_ids[i]].set_result(partial_result)
                    start_idx = end_idx
        except asyncio.TimeoutError:
            self.error_counter += 1
            for req_id in request_ids:
                if req_id in self.response_futures:
                    self.response_futures[req_id].set_exception(
                        TimeoutError("GPU processing timeout")
                    )
        except Exception as e:
            self.error_counter += 1
            logger.error(f"Processing error: {e}")
            for req_id in request_ids:
                if req_id in self.response_futures:
                    self.response_futures[req_id].set_exception(e)
        finally:
            for _ in request_ids:
                self.active_requests -= 1

    async def run_with_semaphore(self, func, data):
        """Run a function with GPU lock and return results."""
        start_time = time.time()
        async with self.gpu_lock:  # Wait for semaphore
            try:
                future = self.executor.submit(func, data)
                result = await asyncio.wait_for(
                    asyncio.wrap_future(future), timeout=Config.GPU_TIMEOUT
                )
                processing_time = time.time() - start_time
                result["processing_time"] = processing_time
                return result
            except asyncio.TimeoutError:
                raise
            except Exception as e:
                raise e

    async def process_request(self, request_data: EmbedRequest):
        """Queue a request for processing and await the result."""
        try:
            # Check if we're at max capacity
            if self.active_requests >= Config.MAX_REQUEST:
                raise HTTPException(
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    detail="Server is currently at maximum capacity. Please try again later.",
                )

            # Process the request
            await self.ensure_processing_loop_started()
            request_id = str(uuid4())
            self.response_futures[request_id] = asyncio.Future()
            self.active_requests += 1
            self.request_counter += 1

            try:
                await asyncio.wait_for(
                    self.queue.put((request_data, request_id)),
                    timeout=1.0,  # Timeout for queue put
                )
            except asyncio.TimeoutError:
                self.active_requests -= 1
                del self.response_futures[request_id]
                raise HTTPException(
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    detail="Request queue is full. Please try again later.",
                )

            try:
                result = await asyncio.wait_for(
                    self.response_futures[request_id], timeout=Config.REQUEST_TIMEOUT
                )
                del self.response_futures[request_id]
                return result
            except asyncio.TimeoutError:
                self.active_requests -= 1
                self.error_counter += 1
                if request_id in self.response_futures:
                    del self.response_futures[request_id]
                raise HTTPException(
                    status_code=HTTP_504_GATEWAY_TIMEOUT,
                    detail="Request processing timed out",
                )

        except HTTPException:
            raise
        except Exception as e:
            self.error_counter += 1
            logger.error(f"Request processing error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Internal Server Error: {str(e)}"
            )

    def get_health_status(self) -> Dict[str, Any]:
        """Get service health information."""
        return {
            "status": "healthy",
            "queue_size": self.queue.qsize(),
            "active_requests": self.active_requests,
            "total_requests": self.request_counter,
            "error_count": self.error_counter,
            "uptime": time.time() - self.start_time,
        }


# --- FastAPI App Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application."""
    # Startup
    logger.info("Initializing model and processor...")
    app.state.model = M3ModelWrapper(
        model_dir=os.path.join(Config.MODEL_DIR, Config.MODEL_NAME),
        onnx_file=Config.ONNX_FILE,
        device=Config.DEVICE,
    )

    app.state.model.warm_up()

    app.state.processor = RequestProcessor(app.state.model)
    logger.info("Server startup complete")

    yield

    # Shutdown
    logger.info("Shutting down...")
    # Clean up any resources
    if hasattr(app.state, "processor") and app.state.processor.executor:
        app.state.processor.executor.shutdown(wait=True)
    logger.info("Server shutdown complete")


app = FastAPI(
    title="Embedding Service",
    description="API for text embeddings using BGE-M3 model",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware if enabled
if Config.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Modify in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Request timing middleware
@app.middleware("http")
async def timeout_and_logging_middleware(request: Request, call_next):
    start_time = time.time()

    # Generate request ID for tracking
    request_id = str(uuid4())
    request.state.request_id = request_id

    path = request.url.path
    method = request.method
    logger.info(f"Request {request_id}: {method} {path} started")

    try:
        response = await asyncio.wait_for(
            call_next(request), timeout=Config.REQUEST_TIMEOUT
        )

        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(
            f"Request {request_id}: {method} {path} completed in {process_time:.3f}s"
        )

        return response

    except asyncio.TimeoutError:
        process_time = time.time() - start_time
        logger.warning(
            f"Request {request_id}: {method} {path} timed out after {process_time:.3f}s"
        )

        return JSONResponse(
            status_code=HTTP_504_GATEWAY_TIMEOUT,
            content={
                "detail": "Request processing time exceeded limit",
                "processing_time": process_time,
            },
        )
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Request {request_id}: {method} {path} failed with error: {str(e)}"
        )

        return JSONResponse(
            status_code=500, content={"detail": f"Internal server error: {str(e)}"}
        )


# Helper function to get processor
def get_processor(request: Request) -> RequestProcessor:
    return request.app.state.processor


# --- API Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check(processor: RequestProcessor = Depends(get_processor)):
    """Check the health status of the service."""
    health_data = processor.get_health_status()
    return health_data


@app.post("/embed/", response_model=EmbedResponse)
async def get_embeddings(
    request: EmbedRequest, processor: RequestProcessor = Depends(get_processor)
):
    """Generate dense and sparse embeddings for a list of sentences."""
    result = await processor.process_request(request)
    return EmbedResponse(
        dense_vecs=result["dense_vecs"],
        lexical_weights=result["lexical_weights"],
        processing_time=result["processing_time"],
    )


# --- Main entrypoint ---
if __name__ == "__main__":
    import uvicorn

    # Print configuration
    logger.info("Starting server with configuration:")
    for key, value in vars(Config).items():
        if not key.startswith("__"):
            logger.info(f"  {key}: {value}")

    uvicorn.run(
        app if Config.WORKERS == 1 else "m3_server:app",
        host=Config.HOST,
        port=Config.PORT,
        workers=Config.WORKERS,
    )
