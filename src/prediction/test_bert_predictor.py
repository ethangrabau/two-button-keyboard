"""
Test suite for evaluating BERT predictor performance.
"""

import torch
import logging
import time
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
from bert_predictor import BertPredictor
from test_data.conversation_test_cases import ALL_TESTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Results for a single test case."""
    expected: str
    predicted: str
    score: float
    latency: float
    description: str
    context: List[str]

@dataclass
class TestSuiteResults:
    """Results for an entire test suite."""
    category: str
    results: List[TestResult]
    avg_score: float
    avg_latency: float
    perfect_matches: int
    total_tests: int

class BertPredictorTester:
    def __init__(self, device: str = "mps"):
        """Initialize the tester with a BERT predictor."""
        self.predictor = BertPredictor(device=device)
        
    def run_single_test(self, test_case: Dict[str, Any]) -> TestResult:
        """Run a single test case and return results."""
        start_time = time.time()
        
        # Get prediction
        predicted = self.predictor.predict(
            context=test_case.get("context", []),
            candidates=test_case["candidates"],
            max_results=1,
            use_cache=False  # Disable cache for testing
        )
        
        latency = time.time() - start_time
        
        if not predicted:
            # Handle prediction failure
            return TestResult(
                expected=test_case["expected"],
                predicted="[FAILED]",
                score=0.0,
                latency=latency,
                description=test_case["description"],
                context=test_case.get("context", [])
            )
            
        # Get first prediction
        top_prediction = predicted[0]
        predicted_text = top_prediction.word
        score = top_prediction.score
        
        return TestResult(
            expected=test_case["expected"],
            predicted=predicted_text,
            score=score,
            latency=latency,
            description=test_case["description"],
            context=test_case.get("context", [])
        )
        
    def run_test_suite(self) -> Dict[str, TestSuiteResults]:
        """Run all test cases and return aggregated results."""
        results = {}
        
        for category, test_cases in ALL_TESTS.items():
            category_results = []
            
            for test_case in test_cases:
                result = self.run_single_test(test_case)
                category_results.append(result)
                
            # Calculate statistics
            scores = [r.score for r in category_results]
            latencies = [r.latency for r in category_results]
            perfect_matches = sum(1 for r in category_results 
                                if r.predicted == r.expected)
                                
            results[category] = TestSuiteResults(
                category=category,
                results=category_results,
                avg_score=np.mean(scores),
                avg_latency=np.mean(latencies),
                perfect_matches=perfect_matches,
                total_tests=len(test_cases)
            )
            
        return results
        
    def print_results(self, results: Dict[str, TestSuiteResults]):
        """Print formatted test results."""
        print("\n=== BERT Predictor Test Results ===\n")
        
        for category, suite_results in results.items():
            print(f"\n{category.upper()} TESTS")
            print("=" * 50)
            print(f"Average Score: {suite_results.avg_score:.3f}")
            print(f"Average Latency: {suite_results.avg_latency*1000:.2f}ms")
            print(f"Perfect Matches: {suite_results.perfect_matches}/{suite_results.total_tests}")
            
            print("\nDetailed Results:")
            for result in suite_results.results:
                print(f"\nTest: {result.description}")
                if result.context:
                    print(f"Context: {' | '.join(result.context)}")
                print(f"Expected: '{result.expected}'")
                print(f"Predicted: '{result.predicted}'")
                print(f"Score: {result.score:.3f}")
                print(f"Latency: {result.latency*1000:.2f}ms")
                
def main():
    """Run the full test suite."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Running tests on device: {device}")
    
    tester = BertPredictorTester(device=device)
    results = tester.run_test_suite()
    tester.print_results(results)

if __name__ == "__main__":
    main()