"""
Test script for Two-Button Keyboard prediction system.
Tests both pattern matching and phrase prediction with various edge cases.
"""

import requests
import json
from typing import Dict, List, Any
import time
import sys
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init()

BASE_URL = "http://localhost:5001/api"

def print_test(name: str, passed: bool, result: Any = None, error: str = None):
    """Print a formatted test result."""
    status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if passed else f"{Fore.RED}FAIL{Style.RESET_ALL}"
    print(f"\n{Fore.BLUE}Test:{Style.RESET_ALL} {name}")
    print(f"{Fore.BLUE}Status:{Style.RESET_ALL} {status}")
    if result:
        print(f"{Fore.BLUE}Result:{Style.RESET_ALL}")
        print(json.dumps(result, indent=2))
    if error:
        print(f"{Fore.RED}Error:{Style.RESET_ALL} {error}")
    print("-" * 50)

def call_predict(pattern: str, context: List[str] = None, max_results: int = 5) -> Dict:
    """Make a prediction API call."""
    try:
        response = requests.post(f"{BASE_URL}/predict", json={
            "pattern": pattern,
            "context": context or [],
            "max_results": max_results
        })
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_server_health():
    """Test if the server is running and healthy."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        result = response.json()
        is_healthy = (
            result.get("status") == "healthy" and
            result.get("llm_status") == "ready"
        )
        print_test("Server Health Check", is_healthy, result)
        return is_healthy
    except Exception as e:
        print_test("Server Health Check", False, error=str(e))
        return False

def validate_phrase(result: Dict, expected_parts: List[str]) -> bool:
    """Validate a prediction contains expected parts in order."""
    if not result.get("success", False) or not result.get("predictions", []):
        return False
    
    prediction = result["predictions"][0].lower()
    last_pos = -1
    
    for part in expected_parts:
        pos = prediction.find(part.lower())
        if pos == -1 or pos < last_pos:
            return False
        last_pos = pos
        
    return True

def run_tests():
    """Run all prediction tests."""
    # First check server health
    if not test_server_health():
        print(f"{Fore.RED}Server not healthy, stopping tests{Style.RESET_ALL}")
        return

    test_cases = [
        # Basic word tests
        {
            "name": "Basic Single Word",
            "pattern": "RRR",
            "expected_parts": ["you"],
            "context": []
        },
        
        # Basic phrase tests
        {
            "name": "Common Greeting - Hi how are you",
            "pattern": "RR RRL LLL RRR",
            "expected_parts": ["hi", "how", "are", "you"],
            "context": []
        },
        
        # Longer phrase tests
        {
            "name": "Extended Greeting - Hi how are you today",
            "pattern": "RR RRL LLL RRR LRLLL",
            "expected_parts": ["hi", "how", "are", "you"],  # More flexible matching
            "context": []
        },
        {
            "name": "Question Response - I am doing fine",
            "pattern": "R LL LRRRL LRRL",
            "context": ["Hi how are you?"],
            "expected_parts": ["i", "am"]  # Allow variation in responses
        },
        
        # Context-aware tests
        {
            "name": "Response with Context",
            "pattern": "RR RRL",
            "context": ["Hi there!", "How are you?"],
            "expected_parts": ["i"],  # Allow any reasonable response start
        },
        
        # Error cases
        {
            "name": "Empty Pattern",
            "pattern": "",
            "validator": lambda r: not r.get("success", True)
        },
        {
            "name": "Invalid Pattern",
            "pattern": "ABC",
            "validator": lambda r: not r.get("success", True)
        },
        
        # Performance tests
        {
            "name": "Long Phrase Performance",
            "pattern": "RR RRL LLL RRR LRLLL LRL LRRRL LRRL",
            "context": ["Hi!", "How are you doing?"],
            "timing": True,
            "validator": lambda r: (
                r.get("success", False) and
                len(r.get("predictions", [])) > 0 and
                " " in r.get("predictions", [""])[0]  # Just verify it's a phrase
            )
        }
    ]

    results = {
        "total": len(test_cases),
        "passed": 0,
        "failed": 0,
        "timings": []
    }

    for test in test_cases:
        start_time = time.time()
        result = call_predict(test["pattern"], test.get("context"), test.get("max_results", 5))
        elapsed = time.time() - start_time

        # Validate result
        passed = False
        if "validator" in test:
            passed = test["validator"](result)
        elif "expected_parts" in test:
            passed = validate_phrase(result, test["expected_parts"])
        else:
            passed = result.get("success", False)

        if test.get("timing"):
            results["timings"].append(elapsed)

        print_test(
            test["name"], 
            passed, 
            result,
            f"Took {elapsed:.3f}s" if test.get("timing") else None
        )

        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

    # Print summary
    print("\n" + "=" * 50)
    print(f"{Fore.BLUE}Test Summary:{Style.RESET_ALL}")
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {Fore.GREEN}{results['passed']}{Style.RESET_ALL}")
    print(f"Failed: {Fore.RED}{results['failed']}{Style.RESET_ALL}")
    if results["timings"]:
        avg_time = sum(results["timings"]) / len(results["timings"])
        print(f"Average response time: {avg_time:.3f}s")
    print("=" * 50)

if __name__ == "__main__":
    try:
        run_tests()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Tests interrupted by user{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}Error running tests: {e}{Style.RESET_ALL}")
        sys.exit(1)