"""
Enhanced Prediction Server for Two-Button Keyboard Interface.
Combines pattern matching with LLM-based selection.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path
import logging
import asyncio
from threading import Thread

# Add the src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from prediction.pattern_matcher import PatternMatcher
from prediction.enhanced_phrase_matcher import EnhancedPhraseMatcher
from llm.phi_interface import Phi2Interface

# Configure logging - only show INFO and above
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set module logging levels
logging.getLogger('prediction.pattern_matcher').setLevel(logging.WARNING)
logging.getLogger('prediction.enhanced_phrase_matcher').setLevel(logging.WARNING)

def is_valid_pattern(pattern: str) -> bool:
    """Check if pattern contains only valid L/R characters."""
    return all(c in ['L', 'R', ' '] for c in pattern.strip())

async def init_llm(llm):
    """Initialize the LLM in background."""
    try:
        success = await llm.initialize()
        logger.info(f"LLM initialization {'successful' if success else 'failed'}")
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Initialize components
    word_matcher = PatternMatcher(str(src_path / "data" / "word_frequencies.json"))
    phrase_matcher = EnhancedPhraseMatcher(word_matcher)
    llm = Phi2Interface()
    
    # Start LLM initialization in background
    def start_llm():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(init_llm(llm))
        loop.close()
    
    Thread(target=start_llm).start()
    
    @app.route('/api/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            pattern = data.get('pattern', '').strip()
            max_results = data.get('max_results', 5)
            context = data.get('context', [])
            
            # Validate pattern
            if not pattern:
                return jsonify({
                    'success': False,
                    'error': 'Empty pattern',
                    'predictions': []
                }), 400
                
            if not is_valid_pattern(pattern):
                return jsonify({
                    'success': False,
                    'error': 'Invalid pattern: must contain only L, R, and spaces',
                    'predictions': []
                }), 400
            
            # Get word candidates for each position
            if ' ' in pattern:
                # Phrase prediction
                positions = phrase_matcher.predict_phrase(
                    pattern_sequence=pattern,
                    context=context,
                    max_candidates=max_results
                )
                
                if not positions:
                    return jsonify({
                        'success': False,
                        'error': 'No valid candidates found',
                        'predictions': []
                    }), 400
                    
                try:
                    # Try LLM-based selection if available
                    if llm.model is not None:
                        selected_words, confidence = llm.select_words(
                            positions=positions,
                            message_history=context
                        )
                        predictions = [" ".join(selected_words)]
                        logger.info(f"Phrase prediction: {predictions[0]} (conf: {confidence})")
                    else:
                        # Fall back to highest frequency candidates
                        predictions = [
                            " ".join(pos.candidates[0].word for pos in positions)
                        ]
                        logger.info(f"Pattern prediction: {predictions[0]}")
                        
                except Exception as e:
                    logger.error(f"LLM selection failed: {e}")
                    predictions = [
                        " ".join(pos.candidates[0].word for pos in positions)
                    ]
                    logger.info(f"Fallback prediction: {predictions[0]}")
                    
            else:
                # Single word prediction
                predictions = word_matcher.predict(pattern, max_results=max_results)
                if not predictions:
                    return jsonify({
                        'success': False,
                        'error': 'No predictions found for pattern',
                        'predictions': []
                    }), 400
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'pattern': pattern,
                'is_phrase': ' ' in pattern
            })
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({
                'success': False,
                'error': str(e),
                'predictions': []
            }), 500

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Check health of prediction services."""
        stats = word_matcher.get_pattern_stats()
        llm_status = "ready" if llm.model is not None else "not initialized"
        
        return jsonify({
            'status': 'healthy',
            'patterns_loaded': stats['total_patterns'],
            'words_loaded': stats['total_words'],
            'llm_status': llm_status,
            'llm_device': llm.device if hasattr(llm, 'device') else None
        })

    @app.route('/api/cleanup', methods=['POST'])
    def cleanup():
        """Clean up LLM resources."""
        try:
            llm.cleanup()
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Starting server on port {port}")
    app = create_app()
    app.run(host='0.0.0.0', port=port, debug=True)