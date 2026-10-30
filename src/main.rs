mod cloak;
mod noise;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use cpal::{
    CallbackInfo, Device, Error, ErrorKind, FromSample, SampleFormat, SizedSample,
    StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use ringbuf::{
    HeapRb,
    traits::{Consumer, Producer, Split},
};
use std::path::PathBuf;

use cloak::{Cloak, db_to_lin};

#[derive(Parser)]
#[command(version, about = "Mix adversarial ASR-perturbation noise into a microphone stream")]
struct Opt {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Parser, Clone)]
struct NoiseOpts {
    /// RMS level of the noise in dBFS (0 = full scale). Higher = stronger, but more audible.
    #[arg(long, default_value_t = -40.0, allow_hyphen_values = true)]
    noise_db: f32,
    /// Use every .wav in this directory instead of the embedded clips.
    #[arg(long)]
    noise_dir: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Cmd {
    /// List audio devices.
    Devices,
    /// Capture from the input device, add noise, write to the output device (point this at a
    /// virtual cable / null sink so other apps can use it as their microphone).
    Live {
        #[arg(short, long)]
        input_device: Option<String>,
        #[arg(short, long)]
        output_device: Option<String>,
        /// Buffering between input and output, in milliseconds.
        #[arg(short, long, default_value_t = 40.0)]
        latency: f32,
        #[command(flatten)]
        noise: NoiseOpts,
    },
    /// Offline: apply the same processing to a wav file (for testing and measuring).
    File {
        input: PathBuf,
        output: PathBuf,
        #[command(flatten)]
        noise: NoiseOpts,
    },
}

fn build_cloak(rate: u32, n: &NoiseOpts) -> Result<Cloak> {
    let clips = match &n.noise_dir {
        Some(d) => noise::clips_from_dir(d)?,
        None => noise::embedded_clips()?,
    };
    let lp = noise::NoiseLoop::build(&clips, rate, db_to_lin(n.noise_db))?;
    eprintln!(
        "noise loop: {} clips, {:.1}s, {:.1} dBFS RMS",
        clips.len(),
        lp.len() as f32 / rate as f32,
        n.noise_db
    );
    Ok(Cloak::new(lp))
}

fn main() -> Result<()> {
    match Opt::parse().cmd {
        Cmd::Devices => list_devices(),
        Cmd::File { input, output, noise } => run_file(&input, &output, &noise),
        Cmd::Live { input_device, output_device, latency, noise } => {
            run_live(input_device, output_device, latency, &noise)
        }
    }
}

fn list_devices() -> Result<()> {
    let host = cpal::default_host();
    for d in host.devices()? {
        let io = match (d.supports_input(), d.supports_output()) {
            (true, true) => "in/out",
            (true, false) => "in",
            (false, true) => "out",
            _ => "-",
        };
        println!("[{io:6}] {}", d.id()?);
    }
    Ok(())
}

fn run_file(input: &PathBuf, output: &PathBuf, n: &NoiseOpts) -> Result<()> {
    let mut r = hound::WavReader::open(input).context("open input")?;
    let spec = r.spec();
    let ch = spec.channels as usize;
    let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => r
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 * scale))
            .collect::<Result<_, _>>()?,
    };
    let mut cloak = build_cloak(spec.sample_rate, n)?;
    let mut w = hound::WavWriter::create(
        output,
        hound::WavSpec {
            channels: 1,
            sample_rate: spec.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for frame in raw.chunks(ch) {
        let mono = frame.iter().sum::<f32>() / ch as f32;
        let y = cloak.process(mono);
        w.write_sample((y * 32767.0).round() as i16)?;
    }
    w.finalize()?;
    Ok(())
}

fn pick(
    host: &cpal::Host,
    id: Option<String>,
    input: bool,
) -> Result<Device> {
    let dev = match id {
        Some(s) => host.device_by_id(&s.parse().map_err(|e| anyhow!("bad device id: {e:?}"))?),
        None if input => host.default_input_device(),
        None => host.default_output_device(),
    };
    dev.ok_or_else(|| anyhow!("{} device not found", if input { "input" } else { "output" }))
}

fn run_live(
    input_device: Option<String>,
    output_device: Option<String>,
    latency_ms: f32,
    n: &NoiseOpts,
) -> Result<()> {
    let host = cpal::default_host();
    let in_dev = pick(&host, input_device, true)?;
    let out_dev = pick(&host, output_device, false)?;
    eprintln!("input : {}", in_dev.id()?);
    eprintln!("output: {}", out_dev.id()?);

    let in_cfg = in_dev.default_input_config()?;
    let out_cfg = out_dev.default_output_config()?;
    let rate = in_cfg.sample_rate();
    let in_fmt = in_cfg.sample_format();
    let out_fmt = out_cfg.sample_format();
    let in_conf = StreamConfig {
        channels: in_cfg.channels(),
        sample_rate: rate,
        buffer_size: cpal::BufferSize::Default,
    };
    // The output must run at the input's rate: the pipeline does no rate conversion on speech.
    let out_conf = StreamConfig {
        channels: out_cfg.channels(),
        sample_rate: rate,
        buffer_size: cpal::BufferSize::Default,
    };

    let cloak = build_cloak(rate, n)?;
    let latency_frames = (latency_ms / 1000.0 * rate as f32) as usize;

    macro_rules! out_dispatch {
        ($i:ty) => {
            match out_fmt {
                SampleFormat::F32 => run::<$i, f32>(&in_dev, &out_dev, in_conf, out_conf, cloak, latency_frames),
                SampleFormat::I16 => run::<$i, i16>(&in_dev, &out_dev, in_conf, out_conf, cloak, latency_frames),
                SampleFormat::I32 => run::<$i, i32>(&in_dev, &out_dev, in_conf, out_conf, cloak, latency_frames),
                f => bail!("unsupported output sample format {f}"),
            }
        };
    }
    match in_fmt {
        SampleFormat::F32 => out_dispatch!(f32),
        SampleFormat::I16 => out_dispatch!(i16),
        SampleFormat::I32 => out_dispatch!(i32),
        f => bail!("unsupported input sample format {f}"),
    }
}

fn run<I, O>(
    in_dev: &Device,
    out_dev: &Device,
    in_conf: StreamConfig,
    out_conf: StreamConfig,
    mut cloak: Cloak,
    latency_frames: usize,
) -> Result<()>
where
    I: SizedSample + Send + 'static,
    O: SizedSample + FromSample<f32> + Send + 'static,
    f32: FromSample<I>,
{
    let in_ch = in_conf.channels as usize;
    let out_ch = out_conf.channels as usize;

    // Mono ring buffer (post-mix) between the two device callbacks.
    let ring = HeapRb::<f32>::new(latency_frames * 4 + 4096);
    let (mut prod, mut cons) = ring.split();
    for _ in 0..latency_frames {
        let _ = prod.try_push(0.0);
    }

    let input_fn = move |data: &[I], _: &CallbackInfo| {
        for frame in data.chunks(in_ch) {
            let mono = frame.iter().map(|s| s.to_sample::<f32>()).sum::<f32>() / in_ch as f32;
            // Real-time safe: no allocation, no locks. Drop on overflow rather than block.
            let _ = prod.try_push(cloak.process(mono));
        }
    };
    let output_fn = move |data: &mut [O], _: &CallbackInfo| {
        for frame in data.chunks_mut(out_ch) {
            let y = cons.try_pop().unwrap_or(0.0);
            frame.fill(O::from_sample(y));
        }
    };

    let in_stream = in_dev.build_input_stream(in_conf, input_fn, err_fn, None)?;
    let out_stream = out_dev
        .build_output_stream(out_conf, output_fn, err_fn, None)
        .context("output device may not support the input device's sample rate")?;
    in_stream.start()?;
    out_stream.start()?;
    eprintln!("running — Ctrl-C to stop");
    loop {
        std::thread::park();
    }
}

fn err_fn(err: Error) {
    match err.kind() {
        ErrorKind::DeviceChanged | ErrorKind::RealtimeDenied => eprintln!("{err}"),
        _ => eprintln!("stream error: {err}"),
    }
}
