"""
Core Phi-2 model interface with optimized prediction handling.
"""

import torch
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from .logging import logger, perf_logger, timer
from .cache import SmartCache

class Phi2Interface:
    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model_name = model_name
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.cache = SmartCache(1000)
        self.batch_size = 4
        self.model = None
        self.tokenizer = None
        self._last_prediction_time = None
        self._common_patterns = set()  # Track frequently used patterns
        self._prewarm_task = None

    async def initialize(self):
        """Enhanced non-blocking model initialization with pre-warming."""
        perf_logger.info("Starting model initialization")
        async with timer("model_initialization"):
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                # Load tokenizer (fast operation)
                self.tokenizer = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: AutoTokenizer.from_pretrained(self.model_name)
                )
                
                # Load model in background
                self.model = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map=self.device,
                        low_cpu_mem_usage=True
                    )
                )
                
                # Optimize model
                await self._optimize_model()
                
                # Start background pre-warming
                self._prewarm_task = asyncio.create_task(self._prewarm_loop())
                
                logger.info(f"Initialized on {self.device} with batch size {self.batch_size}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize: {e}")
                return False

    async def _optimize_model(self):
        """Enhanced model optimizations."""
        if not self.model:
            return
            
        try:
            async with timer("model_optimization"):
                if self.device == "mps":
                    # Optimize for Metal
                    self.model = self.model.half()  # Use FP16
                    if hasattr(self.model.config, 'use_cache'):
                        self.model.config.use_cache = False
                    self.model.config.torch_dtype = torch.float16
                    
                # Set smaller context window
                self.model.config.max_position_embeddings = 128
                
                # Enable memory-efficient attention if available
                if hasattr(self.model, 'enable_mem_efficient_attention'):
                    self.model.enable_mem_efficient_attention()
                
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")

    async def _prewarm_loop(self):
        """Background task to pre-warm predictions for common patterns."""
        while True:
            try:
                await asyncio.sleep(1.0)
                if not self._common_patterns:
                    continue

                # Get patterns that aren't cached
                for pattern in self._common_patterns:
                    if not self._last_prediction_time or \
                       (datetime.now() - self._last_prediction_time).total_seconds() > 2.0:
                        await self._prewarm_pattern(pattern)
                        
            except Exception as e:
                logger.error(f"Pre-warm loop error: {e}")
                await asyncio.sleep(5.0)

    async def _prewarm_pattern(self, pattern: str):
        """Pre-warm cache for a common pattern."""
        try:
            async with timer(f"prewarm_pattern_{pattern}"):
                # Create a minimal prompt for pre-warming
                prompt = self._create_prompt(pattern, ["test"], None)
                
                # Generate with minimal tokens
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=64
                ).to(self.device)
                
                with torch.inference_mode():
                    _ = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.model.generate(
                            **inputs,
                            max_new_tokens=16,
                            num_beams=1,
                            do_sample=False
                        )
                    )
                
        except Exception as e:
            logger.warning(f"Pre-warm failed for pattern {pattern}: {e}")

    async def predict_next_word(
        self, 
        pattern: str,
        candidates: List[str],
        context: Optional[List[str]] = None,
        timeout: float = 0.5
    ) -> str:
        """Enhanced next word prediction with performance tracking."""
        perf_logger.info(f"Starting prediction for pattern: {pattern}")
        perf_logger.info(f"Candidates: {candidates}")
        if context:
            perf_logger.info(f"Context: {context}")
            
        async with timer("total_prediction"):
            start_time = datetime.now()
            
            # Add to common patterns
            self._common_patterns.add(pattern)
            
            # Check cache with normalized key
            cache_key = f"{pattern}:{','.join(candidates)}:{','.join(context or [])}"
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug("Cache hit!")
                return cached

            # Create prompt
            prompt = self._create_prompt(pattern, candidates, context)
            
            try:
                # Get prediction with timeout
                result = await asyncio.wait_for(
                    self._predict_single(prompt),
                    timeout=timeout
                )
                
                # Cache result
                self.cache.put(cache_key, result)
                
                # Update timing
                self._last_prediction_time = (datetime.now() - start_time).total_seconds()
                
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Prediction timed out after {timeout}s")
                return candidates[0] if candidates else ""
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                return candidates[0] if candidates else ""

    async def _predict_single(self, prompt: str) -> str:
        """Enhanced single prediction with better error handling."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized")
            
        try:
            async with timer("single_prediction"):
                # Tokenize with better truncation
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=False
                ).to(self.device)
                
                # Generate with optimized parameters
                with torch.inference_mode():
                    outputs = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.model.generate(
                            **inputs,
                            max_new_tokens=32,
                            num_beams=1,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                            repetition_penalty=1.1,
                            length_penalty=1.0,
                            no_repeat_ngram_size=3
                        )
                    )
                    
                # Decode with cleanup
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return self._extract_prediction(text)
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def _create_prompt(
        self,
        pattern: str,
        candidates: List[str],
        context: Optional[List[str]] = None
    ) -> str:
        """Create an efficient prompt."""
        prompt = f"Options: {', '.join(candidates)}\n"
        if context:
            prompt += f"Context: {' '.join(context[-2:])}\n"
        prompt += "Selected:"
        return prompt

    def _extract_prediction(self, text: str) -> str:
        """Extract prediction from generated text."""
        if "Selected:" in text:
            text = text.split("Selected:")[-1]
        return text.strip().split()[0] if text.strip() else ""

    async def cleanup(self):
        """Enhanced cleanup with background task handling."""
        if self._prewarm_task:
            self._prewarm_task.cancel()
            try:
                await self._prewarm_task
            except asyncio.CancelledError:
                pass
            self._prewarm_task = None
            
        if self.model:
            self.model.to('cpu')
            self.model = None
            
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear caches
        self.cache = SmartCache(1000)
        self._common_patterns.clear()

    def get_metrics(self) -> Dict:
        """Enhanced metrics including detailed performance statistics."""
        cache_metrics = self.cache.get_metrics()
        
        return {
            # System status
            "device": self.device,
            "batch_size": self.batch_size,
            "model_loaded": self.model is not None,
            "prewarm_active": self._prewarm_task is not None,
            
            # Cache statistics
            **cache_metrics,
            
            # Performance metrics
            "last_prediction_time": f"{(self._last_prediction_time or 0)*1000:.1f}ms" if self._last_prediction_time else None,
            "memory_usage": f"{torch.cuda.memory_allocated()/1024**2:.1f}MB" if torch.cuda.is_available() else "N/A",
            
            # Pattern statistics
            "common_patterns_count": len(self._common_patterns)
        }