"""
Build word frequency list from standard corpus.
Uses top ~20k English words with proper frequencies.
"""

import json
from pathlib import Path
import requests
from collections import defaultdict
import re

def download_word_frequencies():
    """Download word frequency data from GitHub."""
    url = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_full.txt"
    response = requests.get(url)
    return response.text.splitlines()

def parse_frequency_line(line):
    """Parse a line of format 'word frequency'."""
    try:
        word, freq = line.split(' ')
        return word.strip(), float(freq)
    except:
        return None, None

def is_valid_word(word):
    """Check if word contains only letters."""
    return bool(re.match(r'^[a-zA-Z]+$', word))

def normalize_frequencies(frequencies):
    """Normalize frequencies to 0-1 range."""
    max_freq = max(frequencies.values())
    return {word: freq/max_freq for word, freq in frequencies.items()}

def main():
    print("Downloading word frequencies...")
    lines = download_word_frequencies()
    
    # Parse and filter words
    frequencies = {}
    for line in lines:
        word, freq = parse_frequency_line(line)
        if word and is_valid_word(word):
            frequencies[word] = freq
    
    # Normalize frequencies
    frequencies = normalize_frequencies(frequencies)
    
    # Sort by frequency and take top 20k
    sorted_words = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:20000]
    final_frequencies = dict(sorted_words)
    
    # Save to file
    output_path = Path(__file__).parent / 'word_frequencies.json'
    with open(output_path, 'w') as f:
        json.dump(final_frequencies, f, indent=2)
    
    print(f"Saved {len(final_frequencies)} words to {output_path}")
    
    # Print some stats
    lengths = defaultdict(int)
    for word in final_frequencies:
        lengths[len(word)] += 1
    
    print("\nWord length distribution:")
    for length in sorted(lengths):
        print(f"{length} letters: {lengths[length]} words")

if __name__ == '__main__':
    main()