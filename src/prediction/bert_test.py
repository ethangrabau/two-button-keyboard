"""
Profile BERT predictor inference performance and behavior.
Tests latency, parallelism, and interruption capabilities.
"""

import asyncio
import torch
import time
import psutil
import os
import logging
from threading import Lock
from bert_predictor import BertPredictor
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTest:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.predictor = None
        self.current_task = None
        self.lock = Lock()
        
    async def initialize(self):
        """Initialize BERT predictor."""
        logger.info(f"Loading BERT predictor on {self.device}")
        self.predictor = BertPredictor(
            model_name="bert-base-uncased",
            device=self.device,
            max_context_length=128,
            temperature=1.0
        )
        return True
        
    async def run_inference(self, prompt: str, task_id: int = None) -> float:
        """Run single inference and measure latency."""
        with self.lock:
            self.current_task = task_id
            
        # Convert prompt to context and candidates format
        if "Select words" in prompt:
            # Parse word selection prompt
            parts = prompt.split(":")[1:]
            candidates = []
            for part in parts:
                if not part.strip():
                    continue
                words = part.split(",")
                candidates.append([w.strip() for w in words])
            context = []
        else:
            # Use prompt as context
            context = [prompt]
            candidates = [["the", "a", "an"], ["quick", "fast", "swift"]]
            
        start = time.perf_counter()
        
        with torch.inference_mode():
            if self.current_task != task_id:
                logger.info(f"Task {task_id} interrupted!")
                return -1
                
            results = self.predictor.predict(
                context=context,
                candidates=candidates,
                max_results=1,
                use_cache=False  # Disable cache for testing
            )
            
        if self.current_task != task_id:
            logger.info(f"Task {task_id} interrupted!")
            return -1
            
        latency = time.perf_counter() - start
        return latency

    def get_memory_usage(self):
        """Get current memory usage."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

async def main():
    test = PerformanceTest()
    await test.initialize()
    
    # Test 1: Basic Latency
    logger.info("\n=== Test 1: Basic Latency ===")
    test_prompts = [
        "Short test.",  # Very short
        "Test prompt for basic latency measurement",  # Medium
        "Test prompt " * 10,  # Long
        "Select words to form a natural phrase: 1: hi, hello 2: are, were 3: you, they",  # Actual use case
    ]
    
    for i, prompt in enumerate(test_prompts):
        latencies = []
        logger.info(f"\nTesting prompt {i+1} ({len(prompt)} chars):")
        logger.info(f"'{prompt}'")
        for j in range(3):
            latency = await test.run_inference(prompt, i*10 + j)
            latencies.append(latency)
            logger.info(f"  Run {j+1} latency: {latency:.3f}s")
        avg_latency = sum(latencies) / len(latencies)
        logger.info(f"  Average latency: {avg_latency:.3f}s")
    
    # Test 2: Parallel Inference Attempt
    logger.info("\n=== Test 2: Parallel Inference ===")
    async def parallel_test():
        tasks = []
        for i in range(3):
            prompt = f"Parallel test prompt {i}"
            tasks.append(asyncio.create_task(test.run_inference(prompt, i+100)))
        results = await asyncio.gather(*tasks)
        return results
    
    parallel_latencies = await parallel_test()
    logger.info(f"Parallel inference latencies: {[f'{l:.3f}s' for l in parallel_latencies]}")
    
    # Test 3: Interruption
    logger.info("\n=== Test 3: Interruption ===")
    async def interrupt_test():
        # Start long inference
        task1 = asyncio.create_task(test.run_inference("Long test prompt "*50, 200))
        await asyncio.sleep(0.1)  # Let it start
        
        # Try to interrupt with new inference
        task2 = asyncio.create_task(test.run_inference("Interrupting prompt", 201))
        results = await asyncio.gather(task1, task2, return_exceptions=True)
        return results
        
    interrupt_results = await interrupt_test()
    logger.info(f"Interruption test results: {interrupt_results}")
    
    # Memory usage
    logger.info(f"\nFinal memory usage: {test.get_memory_usage():.1f} MB")

if __name__ == "__main__":
    asyncio.run(main())