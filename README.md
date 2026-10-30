# voice-cloak

Mixes adversarial ASR-perturbation audio into a live microphone stream, using
[cpal](https://github.com/RustAudio/cpal) (pinned git dependency, v0.19 — needs Rust 1.85+).

## Usage

    cargo run --release -- devices
    cargo run --release -- live -i <mic id> -o <virtual cable / null sink id> --noise-db=-40
    cargo run --release -- file in.wav out.wav --noise-db=-40     # offline test

Apps use the virtual cable's *output side* as their microphone (BlackHole / VB-Cable on
macOS/Windows; a PipeWire/Pulse null-sink + monitor on Linux). `--noise-dir DIR` loads your own
perturbation clips instead of the embedded ones. On Linux, install `libasound2-dev` first.

## Limitations

The bundled perturbations were optimized for specific utterances against Kaldi. Played over
other speech they are mostly faint noise and are unlikely to stop modern ASR (e.g. Whisper).
Measure word error rate with the `file` command before relying on this for privacy.

## Attribution / license

`assets/*_diff.wav` are from https://github.com/rub-ksv/adversarialattacks (Schönherr et al.,
"Adversarial Attacks Against ASR Systems via Psychoacoustic Hiding", NDSS 2019). Their license
(`assets/LICENSE-adversarialattacks.md`) permits use for science and research **without commercial
interests** — it applies to those clips. Choose a license for the rest of this code yourself.
