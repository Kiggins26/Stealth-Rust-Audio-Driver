//! Hardware-independent mixing core.

use crate::noise::NoiseLoop;

pub struct Cloak {
    noise: NoiseLoop,
}

impl Cloak {
    pub fn new(noise: NoiseLoop) -> Self {
        Self { noise }
    }

    /// Mix one mono sample of speech with the next noise sample. The output is hard-limited to
    /// [-1, 1] so a hot mic cannot wrap around when converted to an integer sample format.
    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        (x + self.noise.next()).clamp(-1.0, 1.0)
    }
}

pub fn db_to_lin(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::{NoiseLoop, embedded_clips};

    #[test]
    fn clamps_and_adds() {
        let n = NoiseLoop::build(&embedded_clips().unwrap(), 16_000, 0.05).unwrap();
        let mut c = Cloak::new(n);
        assert!(c.process(2.0) <= 1.0);
        assert!(c.process(-2.0) >= -1.0);
        assert!((db_to_lin(-20.0) - 0.1).abs() < 1e-6);
    }
}
