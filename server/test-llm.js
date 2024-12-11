import { LLMManager } from '../dist/server/llm/config.js';

async function testPredictions() {
  console.log("Starting basic test...");
  const llm = new LLMManager();
  
  try {
    console.log("\nInitializing LLM...");
    await llm.initialize();
    
    const testCases = [
      // Single word
      "the",
      // Simple two-word phrases
      "I am",
      "they were",
      // Basic context
      "the dog is",
      "I want to"
    ];
    
    console.log("\nRunning predictions...");
    console.log("===================");
    
    for (const text of testCases) {
      try {
        console.log(`\nTesting: "${text}"`);
        const result = await llm.predictNextWord(text);
        console.log(`Prediction: "${result}"\n---`);
      } catch (error) {
        console.error(`Error for "${text}":`, error);
      }
      // Delay between tests
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    
  } catch (error) {
    console.error("Test failed:", error);
  } finally {
    console.log("\nCleaning up...");
    await llm.cleanup();
    console.log("Test complete!");
  }
}

console.log("LLM Prediction Test");
console.log("==================");
testPredictions().catch(error => {
  console.error("Uncaught error:", error);
  process.exit(1);
});