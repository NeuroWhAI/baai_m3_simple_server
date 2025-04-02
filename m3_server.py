from FlagEmbedding import BGEM3FlagModel
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("embedding-service")

# Configuration - moved to environment variables with sensible defaults
class Config:
    # Model settings
    MODEL_NAME = os.environ.get("MODEL_NAME", "BAAI/bge-m3")
    DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    USE_FP16 = os.environ.get("USE_FP16", "True").lower() in ("true", "1", "yes") and DEVICE != "cpu"
    
    # Processing settings
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))  # GPU batch size based on VRAM
    MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "5000"))  # Max context length for embeddings
    
    # Queue and timeout settings
    MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "100"))
    MAX_REQUEST = int(os.environ.get("MAX_REQUEST", "10"))  # Max pending requests
    REQUEST_FLUSH_TIMEOUT = float(os.environ.get("REQUEST_FLUSH_TIMEOUT", "0.05"))  # Seconds
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30"))  # Seconds
    GPU_TIMEOUT = int(os.environ.get("GPU_TIMEOUT", "60"))  # Seconds
    
    # Server settings
    HOST = os.environ.get("HOST", "localhost")
    PORT = int(os.environ.get("PORT", "3000"))
    WORKERS = int(os.environ.get("WORKERS", "1"))  # Number of worker processes
    ENABLE_CORS = os.environ.get("ENABLE_CORS", "False").lower() in ("true", "1", "yes")
    
    # Worker threads for the ThreadPoolExecutor
    WORKER_THREADS = int(os.environ.get("WORKER_THREADS", "4"))


class M3ModelWrapper:
    """Wrapper for the BGEM3FlagModel to handle embedding operations."""
    def __init__(self, model_name: str, device: str = 'cuda', use_fp16: bool = True):
        logger.info(f"Initializing model {model_name} on {device} (FP16: {use_fp16})")
        try:
            self.model = BGEM3FlagModel(model_name, device=device, use_fp16=use_fp16)
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
            result = self.model.encode(
                sentences, 
                batch_size=Config.BATCH_SIZE,
                max_length=Config.MAX_LENGTH,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            
            # Extract both dense vectors and lexical weights (sparse vectors)
            dense_vecs = result['dense_vecs'].tolist()
            lexical_weights = result['lexical_weights']
            
            processing_time = time.time() - start_time
            logger.debug(f"Embedding {len(sentences)} sentences took {processing_time:.2f}s")
            
            return {
                'dense_vecs': dense_vecs,
                'lexical_weights': lexical_weights,
                'processing_time': processing_time
            }
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise


# --- Pydantic Models ---
class EmbedRequest(BaseModel):
    sentences: List[str] = Field(..., min_items=1, description="List of sentences to embed")


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

    async def ensure_processing_loop_started(self):
        """Ensures the request processing loop is running."""
        if not self.processing_loop_started:
            logger.info('Starting processing loop')
            self.processing_loop_task = asyncio.create_task(self.processing_loop())
            self.processing_loop_started = True

    async def processing_loop(self):
        """Main processing loop that handles batching of requests."""
        while True:
            try:
                requests, request_ids = [], []
                start_time = asyncio.get_event_loop().time()

                # Collect requests until batch is full or timeout occurs
                while len(requests) < Config.MAX_REQUEST:
                    timeout = Config.REQUEST_FLUSH_TIMEOUT - (asyncio.get_event_loop().time() - start_time)
                    if timeout <= 0:
                        break

                    try:
                        req_data, req_id = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        requests.append(req_data)
                        request_ids.append(req_id)
                    except asyncio.TimeoutError:
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
        indices = []
        for idx, req in enumerate(requests):
            for sentence in req.sentences:
                all_sentences.append(sentence)
                indices.append(idx)
        
        # Process the combined batch
        embed_task = asyncio.create_task(self.run_with_semaphore(
            self.model.embed, 
            all_sentences,
            request_ids,
            request_sizes=[len(req.sentences) for req in requests]
        ))
        
        await embed_task

    async def run_with_semaphore(self, func, data, request_ids, request_sizes):
        """Run a function with GPU lock and handle results."""
        start_time = time.time()
        async with self.gpu_lock:  # Wait for semaphore
            try:
                future = self.executor.submit(func, data)
                result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=Config.GPU_TIMEOUT)
                processing_time = time.time() - start_time
                
                # Split the results according to the original request sizes
                start_idx = 0
                for i, size in enumerate(request_sizes):
                    if i < len(request_ids):
                        end_idx = start_idx + size
                        
                        # Extract portions of both dense and lexical vectors
                        partial_result = {
                            'dense_vecs': result['dense_vecs'][start_idx:end_idx],
                            'lexical_weights': result['lexical_weights'][start_idx:end_idx] ,
                            'processing_time': processing_time
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
                for req_id in request_ids:
                    self.active_requests -= 1

    async def process_request(self, request_data: EmbedRequest):
        """Queue a request for processing and await the result."""
        try:
            # Check if we're at max capacity
            if self.active_requests >= Config.MAX_REQUEST:
                raise HTTPException(
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    detail="Server is currently at maximum capacity. Please try again later."
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
                    timeout=1.0  # Timeout for queue put
                )
            except asyncio.TimeoutError:
                self.active_requests -= 1
                del self.response_futures[request_id]
                raise HTTPException(
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    detail="Request queue is full. Please try again later."
                )
            
            try:
                result = await asyncio.wait_for(
                    self.response_futures[request_id],
                    timeout=Config.REQUEST_TIMEOUT
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
                    detail="Request processing timed out"
                )
            
        except HTTPException:
            raise
        except Exception as e:
            self.error_counter += 1
            logger.error(f"Request processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get service health information."""
        return {
            "status": "healthy",
            "queue_size": self.queue.qsize(),
            "active_requests": self.active_requests,
            "total_requests": self.request_counter,
            "error_count": self.error_counter,
            "uptime": time.time() - self.start_time
        }


# --- FastAPI App Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application."""
    # Startup
    logger.info("Initializing model and processor...")
    app.state.model = M3ModelWrapper(
        model_name=Config.MODEL_NAME,
        device=Config.DEVICE,
        use_fp16=Config.USE_FP16
    )

    app.state.model.warm_up()

    app.state.processor = RequestProcessor(app.state.model)
    logger.info("Server startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    # Clean up any resources
    if hasattr(app.state, 'processor') and app.state.processor.executor:
        app.state.processor.executor.shutdown(wait=True)
    logger.info("Server shutdown complete")


app = FastAPI(
    title="Embedding Service",
    description="API for text embeddings using BGE-M3 model",
    version="1.0.0",
    lifespan=lifespan
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
            call_next(request), 
            timeout=Config.REQUEST_TIMEOUT
        )
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"Request {request_id}: {method} {path} completed in {process_time:.3f}s")
        
        return response
        
    except asyncio.TimeoutError:
        process_time = time.time() - start_time
        logger.warning(f"Request {request_id}: {method} {path} timed out after {process_time:.3f}s")
        
        return JSONResponse(
            status_code=HTTP_504_GATEWAY_TIMEOUT,
            content={
                "detail": "Request processing time exceeded limit",
                "processing_time": process_time
            }
        )
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request {request_id}: {method} {path} failed with error: {str(e)}")
        
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"}
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
    request: EmbedRequest,
    processor: RequestProcessor = Depends(get_processor)
):
    """Generate dense and sparse embeddings for a list of sentences."""
    result = await processor.process_request(request)
    return EmbedResponse(
        dense_vecs=result['dense_vecs'],
        lexical_weights=result['lexical_weights'],
        processing_time=result['processing_time']
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
        workers=Config.WORKERS
    )
