"""
Common phrase patterns and utilities.
"""

COMMON_PHRASES = {
    "RR RRL LLL RRR": [("hi how are you", 1.0)],
    "RLRRR RRL LLL RRR": [("hello how are you", 0.95)],
    "RLR RRL LLL RRR": [("hey how are you", 0.9)],
    "RRL LLL RRR LRRRL": [("how are you doing", 0.9)],
    "LRLL RL LRL": [("what is the", 0.85)],
    "RR LRRRL": [("hi there", 0.8)],
    "RLRRR LRRRL": [("hello there", 0.8)]
}

CONTEXT_WEIGHTS = {
    'frequency': 0.35,    # Reduced slightly to favor context
    'pattern': 0.25,      # Reduced to give more weight to context
    'context': 0.40       # Increased for better phrase coherence
}