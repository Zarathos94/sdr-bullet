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

/// Recovers RDS from a demodulated FM multiplex.
///
/// Feed it the multiplex and the stereo pilot's recovered phase. The subcarrier sits at
/// exactly three times that phase, so no separate carrier recovery is needed — which also
/// means RDS only decodes when the pilot loop is locked.
#[derive(Debug)]
pub struct RdsDecoder {
    subcarrier_filter: Fir,
    baseband_filter: Fir,
    /// Fractional position within the current bit, advancing by `clock_step` each sample.
    clock: f32,
    clock_step: f32,
    /// Running integral of the first and second halves of the current symbol.
    first_half: f32,
    second_half: f32,
    /// Previous biphase decision, for the differential decode.
    previous_bit: Option<bool>,
    /// Sliding 26-bit window over the recovered bitstream.
    window: u32,
    window_fill: u32,
    synchronised: bool,
    /// Which block of the group is expected next.
    block_index: usize,
    group: [u16; 4],
    /// Consecutive failed blocks before sync is abandoned.
    misses: u32,
    decoder: GroupDecoder,
    filtered: Vec<f32>,
    baseband: Vec<f32>,
    shaped: Vec<f32>,
    /// Phase added to the recovered subcarrier reference.
    ///
    /// The standard permits the data subcarrier to sit either in phase or in quadrature
    /// with the pilot's third harmonic, and stations differ. Differential decoding absorbs
    /// a half-turn but not a quarter, so this stays adjustable and is settled against real
    /// transmissions rather than assumed.
    subcarrier_phase: f32,
}

impl RdsDecoder {
    /// # Panics
    /// If the sample rate cannot carry the 57 kHz subcarrier and its sidebands.
    pub fn new(sample_rate: f32) -> Self {
        assert!(
            sample_rate > 122_000.0,
            "RDS needs headroom above the 57 kHz subcarrier, got {sample_rate}"
        );
        let norm = |hz: f32| hz / sample_rate;
        Self {
            // The biphase spectrum spans roughly 2.4 kHz either side of the subcarrier.
            subcarrier_filter: Fir::new(design_bandpass(
                151,
                norm(54_600.0),
                norm(59_400.0),
                Window::kaiser(8.6),
            )),
            baseband_filter: Fir::new(design_lowpass(101, norm(2_400.0), Window::kaiser(8.6))),
            clock: 0.0,
            clock_step: BIT_RATE / sample_rate,
            first_half: 0.0,
            second_half: 0.0,
            previous_bit: None,
            window: 0,
            window_fill: 0,
            synchronised: false,
            block_index: 0,
            group: [0; 4],
            misses: 0,
            decoder: GroupDecoder::new(),
            filtered: Vec::new(),
            baseband: Vec::new(),
            shaped: Vec::new(),
            subcarrier_phase: 0.0,
        }
    }

    /// Rotates the recovered subcarrier reference, in radians.
    pub fn set_subcarrier_phase(&mut self, radians: f32) {
        self.subcarrier_phase = radians;
    }

    pub fn state(&self) -> &RdsState {
        self.decoder.state()
    }

    pub fn is_synchronised(&self) -> bool {
        self.synchronised
    }

    pub fn reset(&mut self) {
        self.subcarrier_filter.reset();
        self.baseband_filter.reset();
        self.clock = 0.0;
        self.first_half = 0.0;
        self.second_half = 0.0;
        self.previous_bit = None;
        self.window = 0;
        self.window_fill = 0;
        self.synchronised = false;
        self.block_index = 0;
        self.misses = 0;
        self.decoder.reset();
    }

    /// Processes a block of multiplex alongside the pilot phase recovered for each sample.
    ///
    /// # Panics
    /// If the two slices differ in length.
    pub fn process(&mut self, mpx: &[f32], pilot_phase: &[f32]) {
        assert_eq!(
            mpx.len(),
            pilot_phase.len(),
            "phase must accompany every sample"
        );
        let n = mpx.len();
        self.filtered.resize(n, 0.0);
        self.baseband.resize(n, 0.0);
        self.shaped.resize(n, 0.0);

        self.subcarrier_filter.process(mpx, &mut self.filtered);

        // Coherent detection against three times the pilot phase. The subcarrier is
        // suppressed, so this is the only phase reference available.
        for k in 0..n {
            self.baseband[k] =
                self.filtered[k] * (3.0 * pilot_phase[k] + self.subcarrier_phase).cos();
        }

        self.baseband_filter
            .process(&self.baseband, &mut self.shaped);

        for k in 0..n {
            self.advance_clock(self.shaped[k]);
        }
    }

    /// Integrates one sample into the current symbol and emits a bit at each boundary.
    fn advance_clock(&mut self, sample: f32) {
        if self.clock < 0.5 {
            self.first_half += sample;
        } else {
            self.second_half += sample;
        }

        self.clock += self.clock_step;
        if self.clock < 1.0 {
            return;
        }
        self.clock -= 1.0;

        // A biphase symbol is a pulse followed by its inverse, so the difference between
        // the two half-integrals carries the bit and any constant offset cancels.
        let decision = self.first_half - self.second_half;
        let bit = decision > 0.0;
        self.first_half = 0.0;
        self.second_half = 0.0;

        // The data is differentially encoded, so a bit is the change between consecutive
        // symbols rather than the symbol itself. That makes it immune to the demodulator
        // settling on the opposite phase.
        let data_bit = match self.previous_bit {
            Some(prev) => prev != bit,
            None => {
                self.previous_bit = Some(bit);
                return;
            }
        };
        self.previous_bit = Some(bit);

        self.push_bit(data_bit);
    }

    fn push_bit(&mut self, bit: bool) {
        self.window = ((self.window << 1) | bit as u32) & 0x3FF_FFFF;
        self.window_fill = (self.window_fill + 1).min(26);
        if self.window_fill < 26 {
            return;
        }

        if self.synchronised {
            // Once synchronised the blocks arrive on a fixed 26-bit cadence, so only test
            // at those boundaries rather than at every bit.
            self.window_fill = 0;
            self.consume_block();
        } else if let Some(offset) = classify(self.window) {
            // The stream carries no framing, so sync means finding a bit position where
            // the syndrome matches. Waiting for an A block gives a known group boundary.
            if offset == Offset::A {
                self.synchronised = true;
                self.block_index = 0;
                self.misses = 0;
                self.window_fill = 0;
                self.consume_block();
            }
        }
    }

    fn consume_block(&mut self) {
        let info = ((self.window >> 10) & 0xFFFF) as u16;
        let expected = match self.block_index {
            0 => Offset::A,
            1 => Offset::B,
            2 => Offset::C,
            _ => Offset::D,
        };

        let actual = classify(self.window);
        // Version B groups substitute C' for C, so accept either in that position.
        let ok = match (self.block_index, actual) {
            (2, Some(Offset::C)) | (2, Some(Offset::CPrime)) => true,
            (_, Some(o)) => o == expected,
            (_, None) => false,
        };

        if ok {
            self.decoder.state.blocks_valid += 1;
            self.misses = 0;
        } else {
            self.decoder.state.blocks_invalid += 1;
            self.misses += 1;
            // Persistent failures mean the bit phase has slipped; drop back to searching.
            if self.misses > 8 {
                self.synchronised = false;
                self.block_index = 0;
                self.window_fill = 0;
                return;
            }
        }

        self.group[self.block_index] = info;
        self.block_index += 1;

        if self.block_index == 4 {
            self.block_index = 0;
            // Only interpret a group whose blocks all checked out. A corrupted block here
            // would put wrong characters into the station name, and those persist on
            // screen far longer than the error that produced them.
            if self.misses == 0 {
                let group = self.group;
                self.decoder.push_group(group);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_offset_word_is_its_own_syndrome() {
        // Each offset word is ten bits, so as a polynomial its degree is below the
        // generator's and dividing leaves it unchanged. Combined with linearity that is
        // what makes the scheme work: a clean block's remainder is exactly the offset word
        // it was scrambled with, so identifying a block is a table lookup.
        //
        // Some decoders publish a different table (0x3D8 for A, and so on). Those come
        // from a register formulation that keeps clocking past the end of the block, not
        // from the standard's plain polynomial remainder. Either is self-consistent, but
        // only this one matches how the check word is defined, so it is the one that
        // agrees with a real broadcast.
        for offset in Offset::ALL {
            assert_eq!(
                expected_syndrome(offset),
                offset.word(),
                "{offset:?} should divide through unchanged"
            );
        }
    }

    #[test]
    fn check_word_construction_matches_the_standard() {
        // The check word is the remainder of the information bits raised by the code's
        // degree, added to the offset. Recomputing it here independently of encode_block
        // guards against the two drifting apart.
        let info = 0x3ABCu16;
        let block = encode_block(info, Offset::B);
        let expected_check = syndrome((info as u32) << 10, 26) ^ Offset::B.word();
        assert_eq!((block & 0x3FF) as u16, expected_check);
        // A clean codeword before scrambling must have a zero remainder.
        let unscrambled = block ^ Offset::B.word() as u32;
        assert_eq!(
            syndrome(unscrambled, 26),
            0,
            "codeword did not divide cleanly"
        );
    }

    #[test]
    fn every_offset_has_a_distinct_syndrome() {
        // If two collided, block identification would be ambiguous.
        let mut seen = Vec::new();
        for o in Offset::ALL {
            let s = expected_syndrome(o);
            assert!(!seen.contains(&s), "syndrome collision on {o:?}");
            seen.push(s);
        }
    }

    #[test]
    fn encoded_blocks_classify_back_to_their_offset() {
        for offset in Offset::ALL {
            for info in [0x0000u16, 0x1234, 0xFFFF, 0xABCD, 0x5555] {
                let block = encode_block(info, offset);
                assert_eq!(
                    classify(block),
                    Some(offset),
                    "info {info:04X} with {offset:?} did not round-trip"
                );
                assert_eq!(
                    ((block >> 10) & 0xFFFF) as u16,
                    info,
                    "information bits altered"
                );
            }
        }
    }

    #[test]
    fn a_single_bit_error_is_detected() {
        let block = encode_block(0x1234, Offset::A);
        for bit in 0..26 {
            let corrupted = block ^ (1 << bit);
            assert_ne!(
                classify(corrupted),
                Some(Offset::A),
                "flipping bit {bit} went undetected"
            );
        }
    }

    #[test]
    fn syndrome_is_linear() {
        // Linearity is what makes offset scrambling recoverable at all.
        let a = 0x0123_4567u32 & 0x3FF_FFFF;
        let b = 0x0089_ABCDu32 & 0x3FF_FFFF;
        assert_eq!(syndrome(a ^ b, 26), syndrome(a, 26) ^ syndrome(b, 26));
    }

    /// Assembles a block B word from its named fields, in the standard's bit layout.
    fn block_b(group_type: u16, version_b: bool, traffic: bool, pty: u16, low5: u16) -> u16 {
        (group_type << 12)
            | ((version_b as u16) << 11)
            | ((traffic as u16) << 10)
            | (pty << 5)
            | low5
    }

    /// Builds the four blocks of a group-0A frame carrying two station-name characters.
    fn name_group(pi: u16, segment: u16, chars: [u8; 2]) -> [u16; 4] {
        let b = block_b(0, false, false, 10, segment);
        [pi, b, 0, ((chars[0] as u16) << 8) | chars[1] as u16]
    }

    #[test]
    fn assembles_a_station_name_from_four_fragments() {
        let mut d = GroupDecoder::new();
        for (segment, pair) in [*b"RA", *b"DI", *b"O ", *b"FM"].iter().enumerate() {
            let chars = [pair[0], pair[1]];
            d.push_group(name_group(0x1234, segment as u16, chars));
        }
        assert_eq!(d.state().station_name, "RADIO FM");
        assert_eq!(d.state().program_id, Some(0x1234));
        assert_eq!(d.state().program_type, Some(10));
    }

    #[test]
    fn withholds_the_name_until_every_fragment_has_arrived() {
        let mut d = GroupDecoder::new();
        d.push_group(name_group(0x1234, 0, *b"AB"));
        d.push_group(name_group(0x1234, 1, *b"CD"));
        assert_eq!(d.state().station_name, "", "published a half-built name");

        d.push_group(name_group(0x1234, 2, *b"EF"));
        d.push_group(name_group(0x1234, 3, *b"GH"));
        assert_eq!(d.state().station_name, "ABCDEFGH");
    }

    #[test]
    fn fragments_may_arrive_out_of_order() {
        let mut d = GroupDecoder::new();
        for (segment, pair) in [(3u16, *b"GH"), (1, *b"CD"), (0, *b"AB"), (2, *b"EF")] {
            d.push_group(name_group(0x1234, segment, [pair[0], pair[1]]));
        }
        assert_eq!(d.state().station_name, "ABCDEFGH");
    }

    /// Builds a group-2A frame carrying four characters of radio text.
    fn text_group(pi: u16, segment: u16, flag: bool, chars: [u8; 4]) -> [u16; 4] {
        let b = block_b(2, false, false, 10, ((flag as u16) << 4) | segment);
        [
            pi,
            b,
            ((chars[0] as u16) << 8) | chars[1] as u16,
            ((chars[2] as u16) << 8) | chars[3] as u16,
        ]
    }

    #[test]
    fn assembles_radio_text_across_segments() {
        let mut d = GroupDecoder::new();
        let message = b"Now Playing: Something Long Enough To Span";
        for (segment, chunk) in message.chunks(4).enumerate() {
            let mut chars = [b' '; 4];
            chars[..chunk.len()].copy_from_slice(chunk);
            d.push_group(text_group(0x1234, segment as u16, false, chars));
        }
        assert_eq!(
            d.state().radio_text,
            String::from_utf8_lossy(message).trim_end()
        );
    }

    #[test]
    fn a_toggled_text_flag_discards_the_previous_message() {
        let mut d = GroupDecoder::new();
        d.push_group(text_group(0x1234, 0, false, *b"OLDS"));
        d.push_group(text_group(0x1234, 1, false, *b"TUFF"));
        assert!(d.state().radio_text.starts_with("OLDSTUFF"));

        // The flag flipping means a new message; the old characters must not survive.
        d.push_group(text_group(0x1234, 0, true, *b"NEW!"));
        assert!(
            !d.state().radio_text.contains("STUFF"),
            "stale text survived the flag toggle: {:?}",
            d.state().radio_text
        );
        assert!(d.state().radio_text.starts_with("NEW!"));
    }

    #[test]
    fn a_carriage_return_terminates_the_text() {
        let mut d = GroupDecoder::new();
        d.push_group(text_group(0x1234, 0, false, *b"Hi\r "));
        assert_eq!(d.state().radio_text, "Hi");
    }

    #[test]
    fn control_and_high_bytes_are_replaced_rather_than_shown() {
        let mut d = GroupDecoder::new();
        for (segment, chars) in [
            (0u16, [b'O', 0x00]),
            (1, [b'K', 0xFF]),
            (2, *b"  "),
            (3, *b"  "),
        ] {
            d.push_group(name_group(0x1234, segment, chars));
        }
        let name = &d.state().station_name;
        assert!(
            name.chars().all(|c| c.is_ascii_graphic() || c == ' '),
            "unsanitised: {name:?}"
        );
        assert!(name.starts_with('O'));
    }

    #[test]
    fn traffic_flag_is_read_from_block_b() {
        let mut d = GroupDecoder::new();
        let mut group = name_group(0x1234, 0, *b"AB");
        group[1] |= 1 << 10;
        d.push_group(group);
        assert!(d.state().traffic_program);
    }

    #[test]
    fn unknown_group_types_update_the_common_fields_only() {
        let mut d = GroupDecoder::new();
        // Group type 8 carries traffic message data this decoder does not interpret.
        let b = block_b(8, false, false, 5, 0);
        d.push_group([0xBEEF, b, 0x1111, 0x2222]);

        assert_eq!(d.state().program_id, Some(0xBEEF));
        assert_eq!(d.state().program_type, Some(5));
        assert_eq!(d.state().station_name, "");
        assert_eq!(d.state().groups_decoded, 1);
    }

    #[test]
    fn block_error_rate_reflects_the_counters() {
        let mut s = RdsState::default();
        assert_eq!(s.block_error_rate(), 0.0);
        s.blocks_valid = 90;
        s.blocks_invalid = 10;
        assert!((s.block_error_rate() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn bit_level_sync_locks_onto_an_encoded_stream() {
        // Drive the sliding-window sync directly with a known bitstream, bypassing the
        // signal path. This is the layer that has to find block boundaries with no framing.
        let groups = [
            name_group(0x1234, 0, *b"TE"),
            name_group(0x1234, 1, *b"ST"),
            name_group(0x1234, 2, *b"IN"),
            name_group(0x1234, 3, *b"G!"),
        ];

        let mut decoder = RdsDecoder::new(240_000.0);
        // Lead in with arbitrary bits so sync has to actually search rather than start
        // aligned by luck.
        for k in 0..37 {
            decoder.push_bit(k % 3 == 0);
        }
        // Repeat the sequence so every name fragment is seen after sync is acquired.
        for _ in 0..3 {
            for group in &groups {
                for (idx, info) in group.iter().enumerate() {
                    let offset = match idx {
                        0 => Offset::A,
                        1 => Offset::B,
                        2 => Offset::C,
                        _ => Offset::D,
                    };
                    let block = encode_block(*info, offset);
                    for bit in (0..26).rev() {
                        decoder.push_bit((block >> bit) & 1 == 1);
                    }
                }
            }
        }

        assert!(decoder.is_synchronised(), "never acquired block sync");
        assert_eq!(decoder.state().station_name, "TESTING!");
        assert_eq!(decoder.state().program_id, Some(0x1234));
        assert!(
            decoder.state().blocks_invalid == 0,
            "{} blocks failed their syndrome check",
            decoder.state().blocks_invalid
        );
    }

    #[test]
    #[should_panic(expected = "headroom above the 57 kHz subcarrier")]
    fn rejects_a_rate_that_cannot_carry_the_subcarrier() {
        RdsDecoder::new(48_000.0);
    }
}
