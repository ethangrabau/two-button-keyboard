"""
Test cases for evaluating the BERT predictor's sequence selection capabilities.
"""

NO_CONTEXT_TESTS = [
    {
        "expected": "we could do",
        "candidates": [
            ["we", "i", "they", "you"],
            ["could", "should", "will", "can"],
            ["do", "try", "go", "make"]
        ],
        "description": "Basic action phrase"
    },
    {
        "expected": "sounds good to",
        "candidates": [
            ["looks", "seems", "sounds", "feels"],
            ["great", "good", "fine", "nice"],
            ["to", "for", "and", "but"]
        ],
        "description": "Common agreement phrase"
    },
    {
        "expected": "let me know",
        "candidates": [
            ["let", "have", "just", "please"],
            ["me", "us", "him", "them"],
            ["know", "see", "think", "try"]
        ],
        "description": "Request phrase"
    }
]

CONTEXT_TESTS = [
    {
        "expected": "that works for",
        "candidates": [
            ["this", "that", "it", "which"],
            ["works", "sounds", "looks", "seems"],
            ["for", "with", "to", "and"]
        ],
        "context": ["When should we meet?", "How about Friday at 6?"],
        "description": "Agreement in scheduling context"
    },
    {
        "expected": "we are available",
        "candidates": [
            ["we", "i", "they", "you"],
            ["are", "were", "get", "feel"],
            ["available", "ready", "going", "planning"]
        ],
        "context": ["Can you do next Thursday?"],
        "description": "Availability response"
    },
    {
        "expected": "i can help",
        "candidates": [
            ["i", "we", "they", "you"],
            ["can", "will", "might", "should"],
            ["help", "try", "go", "come"]
        ],
        "context": ["Could someone look at this implementation?"],
        "description": "Offering assistance" 
    }
]

EMOTIONAL_CONTEXT_TESTS = [
    {
        "expected": "so excited for",
        "candidates": [
            ["so", "very", "really", "just"],
            ["excited", "happy", "glad", "ready"],
            ["for", "about", "to", "with"]
        ],
        "context": ["We got tickets to the show!", "It's going to be amazing"],
        "description": "Emotional resonance"
    },
    {
        "expected": "that sounds perfect",
        "candidates": [
            ["that", "this", "it", "which"],
            ["sounds", "looks", "seems", "feels"],
            ["perfect", "great", "good", "amazing"]
        ],
        "context": ["Should we meet at 6:30 for dinner?"],
        "description": "Enthusiastic agreement"
    }
]

COMPLEX_CONTEXT_TESTS = [
    {
        "expected": "see you there",
        "candidates": [
            ["meet", "see", "find", "catch"],
            ["you", "all", "everyone", "us"],
            ["there", "then", "soon", "later"]
        ],
        "context": [
            "We're meeting at Bombay House",
            "I'll be there around 6",
            "Perfect, we can get a table"
        ],
        "description": "Multi-turn conversation closure"
    }
]

# All test cases combined
ALL_TESTS = {
    "no_context": NO_CONTEXT_TESTS,
    "with_context": CONTEXT_TESTS,
    "emotional_context": EMOTIONAL_CONTEXT_TESTS,
    "complex_context": COMPLEX_CONTEXT_TESTS
}
