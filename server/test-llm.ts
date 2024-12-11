import { LLMManager } from './llm/config.js';

async function testLLMParameters() {
  console.log("Starting LLM parameter tests...\n");
  
  const configurations = [
    { temperature: 0.1, topK: 20, topP: 0.9 }  // Starting with just one config for testing
  ];
  
  const testCases = [
    "I want to go",
    "The cat sat on the",
    "She looked up at the"
  ];
  
  for (const config of configurations) {
    console.log(`\nTesting with parameters:`, config);
    const llm = new LLMManager();
    
    try {
      console.log("Initializing LLM...");
      await llm.initialize({
        modelPath: "models/tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf",  // Updated path relative to root
        contextSize: 512,  // Reduced for testing
        batchSize: 512,
        threads: 4
      });
      
      console.log("\nRunning predictions...");
      for (const text of testCases) {
        const startTime = performance.now();
        const prediction = await llm.predictNextWord(text);
        const duration = performance.now() - startTime;
        
        console.log(`\nInput: "${text}"`);
        console.log(`Prediction: "${prediction}"`);
        console.log(`Time: ${duration.toFixed(2)}ms`);
      }
      
    } catch (error) {
      console.error("Test failed:", error);
      console.error("Error details:", error instanceof Error ? error.stack : String(error));
    } finally {
      console.log("\nCleaning up...");
      await llm.cleanup();
    }
  }
}

console.log("LLM Test Script");
console.log("===============");
testLLMParameters().catch(err => {
  console.error("Uncaught error:", err);
  console.error("Error details:", err instanceof Error ? err.stack : String(err));
  process.exit(1);
});