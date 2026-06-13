//! Radio Data System: the 57 kHz data subcarrier carried alongside broadcast FM.
//!
//! Structurally this is three problems stacked. A signal-processing one — recover a
//! 1187.5 bit/s biphase stream from a suppressed-carrier subcarrier at three times the
//! stereo pilot. A coding one — the stream has no framing, so block boundaries have to be
//! found by testing every bit position against a shortened cyclic code until the syndromes
//! line up. And a protocol one — reassembling station name and radio text from two- and
//! four-character fragments that arrive out of order and repeat indefinitely.
//!
//! The subcarrier is locked to the pilot in the transmitter (57 kHz is exactly three times
//! 19 kHz, and the bit rate is exactly a 48th of that), so the pilot loop supplies both the
//! carrier phase and the bit clock's frequency. Only the clock's phase has to be recovered.

use crate::fir::{design_bandpass, design_lowpass, Fir};
use crate::window::Window;

/// Bits per second on the subcarrier. Exactly `57000 / 48`.
pub const BIT_RATE: f32 = 1187.5;

/// Feedback taps of the shortened cyclic code, excluding the leading term.
///
/// The generator is `x^10 + x^8 + x^7 + x^5 + x^4 + x^3 + 1`.
const GENERATOR: u16 = 0x1B9;

/// The five offset words. Each block in a group is scrambled by a different one, which is
/// what lets a receiver work out which of the four it is looking at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Offset {
    A,
    B,
    C,
    /// Used in place of C by version B groups.
    CPrime,
    D,
}

impl Offset {
    fn word(self) -> u16 {
        match self {
            Offset::A => 0x0FC,
            Offset::B => 0x198,
            Offset::C => 0x168,
            Offset::CPrime => 0x350,
            Offset::D => 0x1B4,
        }
    }

    const ALL: [Offset; 5] = [Offset::A, Offset::B, Offset::C, Offset::CPrime, Offset::D];
}

/// Remainder of `vec` (interpreted as `len` bits) under the RDS generator polynomial.
///
/// A correctly received block scrambled by offset word `O` leaves the syndrome of `O`
/// behind, because the code is linear and the data part divides cleanly. That turns block
/// identification into a table lookup rather than a search.
pub fn syndrome(vec: u32, len: u32) -> u16 {
    let mut reg: u16 = 0;
    for i in (0..len).rev() {
        let bit = ((vec >> i) & 1) as u16;
        let feedback = (reg & 0x200) != 0;
        reg = ((reg << 1) & 0x3FF) | bit;
        if feedback {
            reg ^= GENERATOR;
        }
    }
    reg
}

/// Syndrome a valid block carrying the given offset word will produce.
pub fn expected_syndrome(offset: Offset) -> u16 {
    syndrome(offset.word() as u32, 26)
}

/// Identifies which offset word a 26-bit block was scrambled with, if any.
pub fn classify(block: u32) -> Option<Offset> {
    let s = syndrome(block & 0x3FF_FFFF, 26);
    Offset::ALL.into_iter().find(|o| expected_syndrome(*o) == s)
}

/// Builds a valid 26-bit block from 16 information bits and an offset word.
///
/// Only the tests and the synthetic generator need this, but keeping it beside the
/// syndrome code means the two cannot drift apart.
pub fn encode_block(info: u16, offset: Offset) -> u32 {
    // The check word is the remainder of the information bits shifted into the code's
    // degree, then scrambled by the offset.
    let check = syndrome((info as u32) << 10, 26) ^ offset.word();
    ((info as u32) << 10) | (check as u32 & 0x3FF)
}

/// Everything the decoder has learned about the station currently tuned.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RdsState {
    /// Programme identification code, which uniquely identifies the station.
    pub program_id: Option<u16>,
    /// Programme type, an index into a genre table that differs by region.
    pub program_type: Option<u8>,
    /// Station name, eight characters.
    pub station_name: String,
    /// Radio text, up to sixty-four characters.
    pub radio_text: String,
    /// Whether the station carries traffic announcements.
    pub traffic_program: bool,
    pub blocks_valid: u64,
    pub blocks_invalid: u64,
    pub groups_decoded: u64,
}

impl RdsState {
    /// Fraction of blocks that passed their syndrome check. A useful signal-quality proxy.
    pub fn block_error_rate(&self) -> f32 {
        let total = self.blocks_valid + self.blocks_invalid;
        if total == 0 {
            0.0
        } else {
            self.blocks_invalid as f32 / total as f32
        }
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

/// Assembles decoded blocks into groups and interprets them.
///
/// Separated from the signal path so it can be driven directly from known-good blocks,
/// which is the only way to test the protocol layer without also testing the demodulator.
#[derive(Debug, Clone)]
pub struct GroupDecoder {
    /// Station name is built from two-character fragments, so it is held as bytes until
    /// all four fragments have arrived at least once.
    name_buffer: [u8; 8],
    name_seen: u8,
    text_buffer: [u8; 64],
    text_seen: u64,
    /// Radio text restarts when this toggles, which is how a station signals new content.
    text_flag: Option<bool>,
    state: RdsState,
}

impl Default for GroupDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl GroupDecoder {
    pub fn new() -> Self {
        Self {
            name_buffer: [b' '; 8],
            name_seen: 0,
            text_buffer: [b' '; 64],
            text_seen: 0,
            text_flag: None,
            state: RdsState::default(),
        }
    }

    pub fn state(&self) -> &RdsState {
        &self.state
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Feeds one complete group: four blocks of sixteen information bits each.
    pub fn push_group(&mut self, blocks: [u16; 4]) {
        self.state.groups_decoded += 1;

        let pi = blocks[0];
        self.state.program_id = Some(pi);

        let b = blocks[1];
        let group_type = (b >> 12) & 0xF;
        let version_b = (b >> 11) & 1 == 1;
        self.state.traffic_program = (b >> 10) & 1 == 1;
        self.state.program_type = Some(((b >> 5) & 0x1F) as u8);

        match group_type {
            0 => self.decode_station_name(b, blocks[3]),
            2 => self.decode_radio_text(b, version_b, blocks[2], blocks[3]),
            _ => {}
        }
    }

    /// Group type 0 carries the station name two characters at a time, with the low two
    /// bits of block B saying which pair.
    fn decode_station_name(&mut self, b: u16, d: u16) {
        let segment = (b & 0x3) as usize;
        let offset = segment * 2;
        self.name_buffer[offset] = (d >> 8) as u8;
        self.name_buffer[offset + 1] = (d & 0xFF) as u8;
        self.name_seen |= 1 << segment;

        // Publish only once every segment has been seen, so the display never shows a
        // name half-built out of two different stations after retuning.
        if self.name_seen == 0b1111 {
            self.state.station_name = sanitise(&self.name_buffer);
        }
    }

    /// Group type 2 carries radio text: four characters per group in version A, two in
    /// version B.
    fn decode_radio_text(&mut self, b: u16, version_b: bool, c: u16, d: u16) {
        let segment = (b & 0xF) as usize;
        let flag = (b >> 4) & 1 == 1;

        // A toggled flag means the station has started sending different text; keeping the
        // old characters would splice two messages together.
        if self.text_flag != Some(flag) {
            self.text_buffer = [b' '; 64];
            self.text_seen = 0;
            self.text_flag = Some(flag);
        }

        let (chars, base): ([u8; 4], usize) = if version_b {
            ([(d >> 8) as u8, (d & 0xFF) as u8, b' ', b' '], segment * 2)
        } else {
            (
                [
                    (c >> 8) as u8,
                    (c & 0xFF) as u8,
                    (d >> 8) as u8,
                    (d & 0xFF) as u8,
                ],
                segment * 4,
            )
        };
        let count = if version_b { 2 } else { 4 };

        for k in 0..count {
            let idx = base + k;
            if idx < self.text_buffer.len() {
                self.text_buffer[idx] = chars[k];
                self.text_seen |= 1 << idx;
            }
        }

        // A carriage return terminates the message early rather than padding to 64.
        let end = self
            .text_buffer
            .iter()
            .position(|c| *c == 0x0D)
            .unwrap_or(self.text_buffer.len());
        self.state.radio_text = sanitise(&self.text_buffer[..end]);
    }
}

/// Replaces control and non-ASCII bytes with spaces, then trims.
///
/// The RDS character set is not ASCII above 0x7F, and stations transmit whatever they like
/// in the unused positions. Passing that through unfiltered puts control characters into a
/// UI string.
fn sanitise(bytes: &[u8]) -> String {
    let s: String = bytes
        .iter()
        .map(|b| {
            if (0x20..0x7F).contains(b) {
                *b as char
            } else {
                ' '
            }
        })
        .collect();
    s.trim_end().to_string()
}

