import { spawn } from 'child_process';
import path from 'path';

interface ModelConfig {
  maxNewTokens: number;
  temperature: number;
  useMetalAcceleration: boolean;
}

export class Phi2Model {
  private pythonProcess: any;
  private config: ModelConfig;
  private isInitialized: boolean = false;

  constructor(config: ModelConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    // Create Python script for model initialization
    const initScript = `
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Enable Metal acceleration if available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Move model to device
model = model.to(device)

print("Model initialized successfully")
    `.trim();

    // Write initialization script
    const scriptPath = path.join(__dirname, 'init_phi2.py');
    await fs.promises.writeFile(scriptPath, initScript);

    try {
      // Run initialization script
      const process = spawn('python', [scriptPath]);
      
      return new Promise((resolve, reject) => {
        process.stdout.on('data', (data: Buffer) => {
          console.log(data.toString());
        });

        process.stderr.on('data', (data: Buffer) => {
          console.error(data.toString());
        });

        process.on('close', (code: number) => {
          if (code === 0) {
            this.isInitialized = true;
            resolve();
          } else {
            reject(new Error(`Model initialization failed with code ${code}`));
          }
        });
      });
    } catch (error) {
      console.error('Failed to initialize Phi-2 model:', error);
      throw error;
    }
  }

  async predict(pattern: string, context: string[] = []): Promise<string> {
    if (!this.isInitialized) {
      throw new Error('Model not initialized');
    }

    const prompt = this.formatPrompt(pattern, context);
    const scriptPath = path.join(__dirname, 'predict.py');
    
    // Create prediction script
    const predictScript = `
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2").to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

def predict(prompt, max_new_tokens=32, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = """${prompt}"""
print(predict(prompt, max_new_tokens=${this.config.maxNewTokens}, temperature=${this.config.temperature}))
    `.trim();

    await fs.promises.writeFile(scriptPath, predictScript);

    return new Promise((resolve, reject) => {
      const process = spawn('python', [scriptPath]);
      
      let output = '';
      process.stdout.on('data', (data) => {
        output += data.toString();
      });

      process.on('exit', (code) => {
        if (code === 0) {
          resolve(output.trim());
        } else {
          reject(new Error(`Prediction failed with code ${code}`));
        }
      });
    });
  }

  private formatPrompt(pattern: string, context: string[]): string {
    // Format: "Previous words... [PATTERN] L R L -> "
    return `${context.join(' ')} [PATTERN] ${pattern} -> `;
  }
}
