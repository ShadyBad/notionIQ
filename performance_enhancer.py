"""
Performance Enhancement Module for NotionIQ
Implements async processing, streaming, connection pooling, and memory optimization
"""

import asyncio
import gc
import json
import sys
import time
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, TypeVar

import aiohttp
import psutil
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from logger_wrapper import logger

console = Console()

T = TypeVar("T")


@dataclass
class PerformanceMetrics:
    """Track performance metrics"""

    total_requests: int = 0
    total_time: float = 0.0
    avg_response_time: float = 0.0
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    items_processed: int = 0
    items_per_second: float = 0.0
    concurrent_tasks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class ConnectionPool:
    """HTTP/2 connection pool with persistent connections"""

    def __init__(self, max_connections: int = 10, keepalive_timeout: int = 30):
        """Initialize connection pool"""
        self.max_connections = max_connections
        self.keepalive_timeout = keepalive_timeout
        self.connector = None
        self.session = None

    async def __aenter__(self):
        """Enter async context"""
        self.connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            keepalive_timeout=self.keepalive_timeout,
            force_close=False,
        )

        timeout = aiohttp.ClientTimeout(
            total=300, connect=10, sock_connect=10, sock_read=30
        )

        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            headers={
                "User-Agent": "NotionIQ/1.0",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            },
        )

        logger.info(
            f"Connection pool initialized with {self.max_connections} connections"
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context"""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()

    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request using pool"""
        if not self.session:
            raise RuntimeError("Connection pool not initialized")

        async with self.session.request(method, url, **kwargs) as response:
            return await response.json()


class StreamProcessor:
    """Process large datasets with streaming to minimize memory usage"""

    def __init__(self, chunk_size: int = 100):
        """Initialize stream processor"""
        self.chunk_size = chunk_size
        self.buffer = deque()

    async def process_stream(
        self,
        items: AsyncIterator[Any],
        processor: Callable[[Any], Any],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> AsyncIterator[Any]:
        """Process items in streaming fashion"""
        chunk = []
        processed_count = 0

        async for item in items:
            chunk.append(item)

            if len(chunk) >= self.chunk_size:
                # Process chunk
                results = await self._process_chunk(chunk, processor)

                for result in results:
                    yield result

                processed_count += len(chunk)
                if progress_callback:
                    progress_callback(processed_count)

                # Clear chunk and force garbage collection
                chunk.clear()
                if processed_count % 1000 == 0:
                    gc.collect()

        # Process remaining items
        if chunk:
            results = await self._process_chunk(chunk, processor)
            for result in results:
                yield result

    async def _process_chunk(
        self, chunk: List[Any], processor: Callable[[Any], Any]
    ) -> List[Any]:
        """Process a chunk of items"""
        tasks = [processor(item) for item in chunk]

        if all(asyncio.iscoroutinefunction(processor) for _ in range(1)):
            return await asyncio.gather(*tasks)
        else:
            return [await task if asyncio.iscoroutine(task) else task for task in tasks]


class AsyncBatchProcessor:
    """Process items in parallel batches for maximum throughput"""

    def __init__(self, max_concurrent: int = 10, batch_size: int = 50):
        """Initialize batch processor"""
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.metrics = PerformanceMetrics()

    async def process_batch(
        self,
        items: List[Any],
        processor: Callable[[Any], Any],
        show_progress: bool = True,
    ) -> List[Any]:
        """Process items in parallel batches"""
        start_time = time.time()
        results = []

        # Split into batches
        batches = [
            items[i : i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Processing {len(items)} items...", total=len(items)
                )

                # Process batches concurrently
                batch_tasks = []
                for batch in batches:
                    batch_task = self._process_batch_with_semaphore(batch, processor)
                    batch_tasks.append(batch_task)

                # Gather results
                for batch_result in asyncio.as_completed(batch_tasks):
                    batch_data = await batch_result
                    results.extend(batch_data)
                    progress.update(task, advance=len(batch_data))
        else:
            # Process without progress bar
            batch_tasks = [
                self._process_batch_with_semaphore(batch, processor)
                for batch in batches
            ]
            batch_results = await asyncio.gather(*batch_tasks)
            for batch_result in batch_results:
                results.extend(batch_result)

        # Update metrics
        elapsed = time.time() - start_time
        self.metrics.total_requests += len(items)
        self.metrics.total_time += elapsed
        self.metrics.avg_response_time = (
            self.metrics.total_time / self.metrics.total_requests
        )
        self.metrics.items_processed += len(items)
        self.metrics.items_per_second = len(items) / elapsed if elapsed > 0 else 0

        logger.info(
            f"Processed {len(items)} items in {elapsed:.2f}s ({self.metrics.items_per_second:.1f} items/s)"
        )

        return results

    async def _process_batch_with_semaphore(
        self, batch: List[Any], processor: Callable[[Any], Any]
    ) -> List[Any]:
        """Process a batch with semaphore limiting"""
        async with self.semaphore:
            self.metrics.concurrent_tasks = self.max_concurrent - self.semaphore._value

            if asyncio.iscoroutinefunction(processor):
                return await asyncio.gather(*[processor(item) for item in batch])
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                tasks = [loop.run_in_executor(None, processor, item) for item in batch]
                return await asyncio.gather(*tasks)


class MemoryOptimizer:
    """Optimize memory usage for large-scale processing"""

    def __init__(self, max_memory_mb: float = 500):
        """Initialize memory optimizer"""
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def check_memory(self) -> bool:
        """Check if memory usage is within limits"""
        current_memory = self.get_memory_usage()
        return current_memory < self.max_memory_mb

    def optimize_if_needed(self, force: bool = False):
        """Optimize memory if usage is high"""
        current_memory = self.get_memory_usage()

        if force or current_memory > self.max_memory_mb * 0.8:
            logger.info(
                f"Memory optimization triggered (current: {current_memory:.1f}MB)"
            )

            # Force garbage collection
            gc.collect()
            gc.collect()  # Second pass for cyclic references

            # Clear caches
            for func in gc.get_objects():
                if hasattr(func, "cache_clear"):
                    try:
                        func.cache_clear()
                    except:
                        pass

            new_memory = self.get_memory_usage()
            freed = current_memory - new_memory
            logger.info(f"Memory freed: {freed:.1f}MB (now: {new_memory:.1f}MB)")

            return freed

        return 0

    @contextmanager
    def memory_limit(self):
        """Context manager to enforce memory limits"""
        initial = self.get_memory_usage()

        try:
            yield
        finally:
            final = self.get_memory_usage()
            used = final - initial

            if used > self.max_memory_mb * 0.5:
                logger.warning(f"High memory usage detected: {used:.1f}MB")
                self.optimize_if_needed(force=True)


class CacheManager:
    """Advanced caching with LRU and TTL support"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize cache manager"""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.timestamps = {}
        self.access_count = {}
        self.metrics = PerformanceMetrics()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            # Check TTL
            if time.time() - self.timestamps[key] < self.ttl_seconds:
                self.access_count[key] = self.access_count.get(key, 0) + 1
                self.metrics.cache_hits += 1
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.timestamps[key]

        self.metrics.cache_misses += 1
        return None

    def set(self, key: str, value: Any):
        """Set item in cache"""
        # Enforce size limit (LRU eviction)
        if len(self.cache) >= self.max_size:
            # Find least recently used item
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.timestamps[lru_key]
            del self.access_count[lru_key]

        self.cache[key] = value
        self.timestamps[key] = time.time()
        self.access_count[key] = 0

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.timestamps.clear()
        self.access_count.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = (
            self.metrics.cache_hits
            / (self.metrics.cache_hits + self.metrics.cache_misses)
            if (self.metrics.cache_hits + self.metrics.cache_misses) > 0
            else 0
        )

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "hits": self.metrics.cache_hits,
            "misses": self.metrics.cache_misses,
            "most_accessed": sorted(
                self.access_count.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }


class PerformanceMonitor:
    """Monitor and report performance metrics"""

    def __init__(self):
        """Initialize performance monitor"""
        self.start_time = time.time()
        self.metrics = PerformanceMetrics()
        self.memory_optimizer = MemoryOptimizer()

    @asynccontextmanager
    async def measure(self, operation: str):
        """Measure performance of an operation"""
        start_time = time.time()
        start_memory = self.memory_optimizer.get_memory_usage()

        logger.info(f"Starting {operation}")

        try:
            yield
        finally:
            elapsed = time.time() - start_time
            memory_used = self.memory_optimizer.get_memory_usage() - start_memory

            logger.info(
                f"Completed {operation} in {elapsed:.2f}s "
                f"(memory: +{memory_used:.1f}MB)"
            )

            # Update metrics
            self.metrics.total_time += elapsed
            self.metrics.current_memory_mb = self.memory_optimizer.get_memory_usage()
            self.metrics.peak_memory_mb = max(
                self.metrics.peak_memory_mb, self.metrics.current_memory_mb
            )

    def get_report(self) -> Dict[str, Any]:
        """Get performance report"""
        uptime = time.time() - self.start_time

        return {
            "uptime_seconds": uptime,
            "total_requests": self.metrics.total_requests,
            "avg_response_time": self.metrics.avg_response_time,
            "items_processed": self.metrics.items_processed,
            "throughput": self.metrics.items_per_second,
            "memory": {
                "current_mb": self.metrics.current_memory_mb,
                "peak_mb": self.metrics.peak_memory_mb,
                "available_mb": psutil.virtual_memory().available / 1024 / 1024,
            },
            "cache": {
                "hits": self.metrics.cache_hits,
                "misses": self.metrics.cache_misses,
                "hit_rate": (
                    self.metrics.cache_hits
                    / (self.metrics.cache_hits + self.metrics.cache_misses)
                    if (self.metrics.cache_hits + self.metrics.cache_misses) > 0
                    else 0
                ),
            },
        }


# Global instances
cache_manager = CacheManager()
performance_monitor = PerformanceMonitor()


async def parallel_fetch(
    urls: List[str], max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    """Fetch multiple URLs in parallel"""
    async with ConnectionPool(max_connections=max_concurrent) as pool:
        processor = AsyncBatchProcessor(max_concurrent=max_concurrent)

        async def fetch_one(url: str) -> Dict[str, Any]:
            try:
                return await pool.request("GET", url)
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
                return {"error": str(e), "url": url}

        results = await processor.process_batch(urls, fetch_one)
        return results


if __name__ == "__main__":
    # Example usage
    async def test_performance():
        console.print("[bold]Testing Performance Enhancements[/bold]\n")

        # Test async batch processing
        console.print("Testing async batch processing...")
        processor = AsyncBatchProcessor(max_concurrent=5, batch_size=10)

        async def simulate_api_call(item: int) -> Dict[str, Any]:
            await asyncio.sleep(0.1)  # Simulate API latency
            return {"id": item, "processed": True}

        items = list(range(50))

        async with performance_monitor.measure("Batch Processing"):
            results = await processor.process_batch(items, simulate_api_call)

        console.print(f"✅ Processed {len(results)} items")

        # Show performance report
        report = performance_monitor.get_report()
        console.print("\n[bold]Performance Report:[/bold]")
        console.print(json.dumps(report, indent=2))

    # Run test
    asyncio.run(test_performance())
