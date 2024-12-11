class Benchmark {
  constructor() {
    this.latencies = [];
    this.maxSamples = 100;
  }

  recordPrediction(result) {
    if (result.latency) {
      this.latencies.push(result.latency);
      if (this.latencies.length > this.maxSamples) {
        this.latencies.shift();
      }
    }
  }

  getStats() {
    if (this.latencies.length === 0) return null;

    const sum = this.latencies.reduce((a, b) => a + b, 0);
    return {
      averageLatency: sum / this.latencies.length,
      samples: this.latencies.length
    };
  }
}

export const benchmark = new Benchmark();