import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_phi2():
    # Check Metal support
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Built with MPS: {torch.backends.mps.is_built()}")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    model = model.to(device)
    
    system_prompt = """You are a predictive text system for a two-button keyboard designed for brain-computer interfaces.
The keyboard is split into two sides:
- Left side:  QWERT ASDFG ZXCV
- Right side: YUIOP HJKL BNVM

When a user presses buttons in a sequence like "L R L", they want words that can be typed using letters from those sides in that order.
For example:
- "L R L" could suggest "test" (t[L] e[L] s[R] t[L])
- "R L R" could suggest "your" (y[R] o[L] u[R])

Your task is to predict the most likely word given:
1. The button pattern (L/R sequence)
2. The previous words (context)
3. Common English word patterns

Only output the predicted word, no explanation."""

    # Test cases with simpler string handling
    test_cases = [
        {
            "context": "I want to",
            "pattern": "R L R",
            "expected": "your"
        },
        {
            "context": "The cat sat on the",
            "pattern": "L R L",
            "expected": "seat"
        },
        {
            "context": "I went to the store to buy some",
            "pattern": "L L R",
            "expected": "tea"
        }
    ]
    
    print("\nRunning test cases...")
    for case in test_cases:
        prompt = f"{system_prompt}\n\nContext: {case['context']}\nPattern: {case['pattern']}\nPredict:"
        
        print(f"\nContext: {case['context']}")
        print(f"Pattern: {case['pattern']}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.7,
                top_k=10,        # Limit to top 10 most likely tokens
                top_p=0.95,      # Nucleus sampling
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_word = generated_text.split("Predict:")[-1].strip()
        print(f"Predicted: {predicted_word}")
        print(f"Expected:  {case['expected']}")

if __name__ == "__main__":
    test_phi2()
