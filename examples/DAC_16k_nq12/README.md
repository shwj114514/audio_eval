# DAC 16 kHz reconstruction example

This example evaluates one DAC-reconstructed utterance against its original
LibriSpeech `test-clean` audio. The DAC reconstruction uses the 16 kHz model
with 12 quantizers (600 tokens/s, codebook size 1024).

Run from the repository root after activating the project environment:

```bash
bash examples/DAC_16k_nq12/run.sh
```

The result is written to
`examples/DAC_16k_nq12/results/dac_16k_nq12.json`.
