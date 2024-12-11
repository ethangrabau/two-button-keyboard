import { getLlama } from 'node-llama-cpp';

async function testBasicLoad() {
  try {
    console.log("Getting llama instance...");
    const llama = await getLlama();
    console.log("Llama instance created");
    
    console.log("Loading model...");
    const model = await llama.loadModel({
      modelPath: "models/tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf",
      contextSize: 512
    });
    console.log("Model loaded successfully");
    
    // Try basic completion
    console.log("Creating context...");
    const context = await model.createContext({
      batchSize: 512,
      threads: 4
    });
    console.log("Context created");
    
    console.log("Getting sequence...");
    const sequence = await context.getSequence();
    console.log("Sequence retrieved");
    
    console.log("Clean up...");
    await context.free();
    await model.free();
    console.log("Cleanup complete");
    
  } catch (error) {
    console.error("Test failed with error:", error);
    console.error("Error stack:", error.stack);
    process.exit(1);
  }
}

console.log("Starting basic load test...");
testBasicLoad().catch(console.error);