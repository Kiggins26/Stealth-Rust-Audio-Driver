//! Adversarial-noise bank.
//!
//! The three `*_diff.wav` clips come from the "Adversarial Attacks Against ASR Systems via
//! Psychoacoustic Hiding" project (rub-ksv/adversarialattacks, `docs/songs/`). Each is the
//! *difference* between a benign clip and its adversarial version, i.e. the perturbation itself,
//! 16 kHz mono 16-bit. They are embedded in the binary; extra clips can be loaded from disk.

use anyhow::{Context, Result, bail};
use std::f32::consts::PI;
use std::io::Cursor;
use std::path::Path;

const EMBEDDED: [(&str, &[u8]); 3] = [
    ("birds_diff", include_bytes!("../assets/birds_diff.wav")),
    ("music_diff", include_bytes!("../assets/music_diff.wav")),
    ("TV_diff", include_bytes!("../assets/TV_diff.wav")),
];

/// A decoded mono clip at its native sample rate.
pub struct Clip {
    pub name: String,
    pub rate: u32,
    pub samples: Vec<f32>,
}

fn decode<R: std::io::Read>(name: &str, reader: R) -> Result<Clip> {
    let mut wav = hound::WavReader::new(reader).with_context(|| format!("reading {name}"))?;
    let spec = wav.spec();
    let ch = spec.channels as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => wav.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            wav.samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    let samples = interleaved
        .chunks(ch)
        .map(|f| f.iter().sum::<f32>() / ch as f32)
        .collect();
    Ok(Clip { name: name.to_string(), rate: spec.sample_rate, samples })
}

pub fn embedded_clips() -> Result<Vec<Clip>> {
    EMBEDDED.iter().map(|(n, b)| decode(n, Cursor::new(*b))).collect()
}

pub fn clips_from_dir(dir: &Path) -> Result<Vec<Clip>> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav")))
        .collect();
    paths.sort();
    if paths.is_empty() {
        bail!("no .wav files in {}", dir.display());
    }
    paths
        .iter()
        .map(|p| decode(&p.display().to_string(), std::fs::File::open(p)?))
        .collect()
}

pub fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

/// Band-limited resampling with a Hann-windowed sinc. Runs once at start-up, so cost is irrelevant
/// compared with getting the spectrum right (linear interpolation would add spectral images that
/// are not part of the perturbation).
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return x.to_vec();
    }
    const HALF: isize = 24; // taps each side
    let ratio = to as f64 / from as f64;
    let cutoff = (1.0f64).min(ratio); // low-pass at the lower Nyquist
    let n_out = (x.len() as f64 * ratio).floor() as usize;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        let t = i as f64 / ratio; // position in input samples
        let c = t.floor() as isize;
        let mut acc = 0.0f64;
        for k in (c - HALF + 1)..=(c + HALF) {
            if k < 0 || k as usize >= x.len() {
                continue;
            }
            let d = t - k as f64;
            let sinc = if d.abs() < 1e-9 {
                1.0
            } else {
                (PI as f64 * d * cutoff).sin() / (PI as f64 * d)
            };
            let w = 0.5 * (1.0 + (PI as f64 * d / HALF as f64).cos());
            acc += x[k as usize] as f64 * sinc * w;
        }
        out.push(acc as f32);
    }
    out
}

/// Seamlessly looping noise at the device sample rate.
pub struct NoiseLoop {
    buf: Vec<f32>,
    pos: usize,
}

impl NoiseLoop {
    /// Each clip is resampled to `rate`, scaled to the same RMS, concatenated with equal-power
    /// cross-fades (including the wrap-around join), then scaled so the loop's RMS equals
    /// `target_rms` (linear, full-scale = 1.0).
    pub fn build(clips: &[Clip], rate: u32, target_rms: f32) -> Result<Self> {
        let xf = (rate as usize * 50) / 1000; // 50 ms cross-fade
        let mut buf: Vec<f32> = Vec::new();
        for c in clips {
            let mut s = resample(&c.samples, c.rate, rate);
            let r = rms(&s);
            if r < 1e-9 {
                continue;
            }
            s.iter_mut().for_each(|v| *v /= r);
            if s.len() <= 2 * xf {
                bail!("clip {} too short", c.name);
            }
            if buf.is_empty() {
                buf = s;
            } else {
                let start = buf.len() - xf;
                for i in 0..xf {
                    let a = (i as f32 + 0.5) / xf as f32 * PI / 2.0;
                    buf[start + i] = buf[start + i] * a.cos() + s[i] * a.sin();
                }
                buf.extend_from_slice(&s[xf..]);
            }
        }
        if buf.len() <= 2 * xf {
            bail!("no usable noise clips");
        }
        // wrap-around join: blend tail into head, then drop the head so the seam is continuous
        let start = buf.len() - xf;
        for i in 0..xf {
            let a = (i as f32 + 0.5) / xf as f32 * PI / 2.0;
            buf[start + i] = buf[start + i] * a.cos() + buf[i] * a.sin();
        }
        buf.drain(..xf);
        let r = rms(&buf);
        buf.iter_mut().for_each(|v| *v *= target_rms / r);
        Ok(Self { buf, pos: 0 })
    }

    #[inline]
    pub fn next(&mut self) -> f32 {
        let v = self.buf[self.pos];
        self.pos += 1;
        if self.pos == self.buf.len() {
            self.pos = 0;
        }
        v
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_preserves_tone() {
        let from = 16_000u32;
        let to = 48_000u32;
        let f = 1_000.0f32;
        let x: Vec<f32> = (0..from as usize)
            .map(|i| (2.0 * PI * f * i as f32 / from as f32).sin())
            .collect();
        let y = resample(&x, from, to);
        assert_eq!(y.len(), to as usize);
        // compare away from the edges
        let mut err = 0.0f32;
        for i in 2000..(y.len() - 2000) {
            let want = (2.0 * PI * f * i as f32 / to as f32).sin();
            err = err.max((y[i] - want).abs());
        }
        assert!(err < 0.01, "max error {err}");
    }

    #[test]
    fn loop_seam_is_continuous_and_hits_target_rms() {
        let clips = embedded_clips().unwrap();
        let mut n = NoiseLoop::build(&clips, 48_000, 0.03).unwrap();
        assert!((rms(&n.buf) - 0.03).abs() < 1e-4);
        // step across the wrap point; the jump must not exceed the largest in-loop step by much
        let max_step = n.buf.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max);
        let last = n.buf[n.len() - 1];
        let first = n.buf[0];
        assert!((first - last).abs() <= max_step * 1.01);
        let _ = n.next();
    }
}
