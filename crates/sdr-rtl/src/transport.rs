//! The USB operations the driver needs, and nothing else.
//!
//! Keeping this an abstract trait rather than depending on a USB library directly is what
//! lets the same register sequences run in three places: against libusb from a terminal,
//! against WebUSB inside a browser worker, and against a recording harness in the tests.
//! The third is what makes the sequences checkable at all — a bus trace tells you what
//! happened, but only after you already have hardware and a failure to look at.

/// Which way data flows on a control transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    In,
    Out,
}

/// A vendor control transfer.
///
/// The RTL2832U uses request zero for everything and encodes the actual target in
/// `value` and `index`. See [`crate::regs`] for the two different encodings involved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlRequest {
    pub direction: Direction,
    pub value: u16,
    pub index: u16,
}

impl ControlRequest {
    pub fn read(value: u16, index: u16) -> Self {
        Self {
            direction: Direction::In,
            value,
            index,
        }
    }

    pub fn write(value: u16, index: u16) -> Self {
        Self {
            direction: Direction::Out,
            value,
            index,
        }
    }
}

/// Transport-level failures, kept deliberately coarse.
///
/// The driver cannot do anything useful with a distinction between, say, a stall and a
/// timeout, so carrying that detail through every call site would be noise. Implementations
/// keep the specifics in [`TransportError::Io`]'s message for logging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportError {
    /// The device went away mid-conversation.
    Disconnected,
    /// The endpoint stalled, which for this device means it rejected the request.
    Stalled,
    /// Anything else, with whatever the underlying library said.
    Io(String),
}

impl core::fmt::Display for TransportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TransportError::Disconnected => write!(f, "device disconnected"),
            TransportError::Stalled => write!(f, "endpoint stalled"),
            TransportError::Io(msg) => write!(f, "transport error: {msg}"),
        }
    }
}

impl std::error::Error for TransportError {}

/// USB access, as the driver needs it.
///
/// Implementations are not required to be `Send`. The browser transport cannot be — it
/// holds JavaScript handles that are bound to one worker — and requiring it would rule
/// that out for no benefit, since the driver is single-threaded by nature anyway.
#[allow(async_fn_in_trait)]
pub trait Transport {
    /// Sends a control transfer carrying `data`.
    async fn control_out(
        &mut self,
        request: ControlRequest,
        data: &[u8],
    ) -> Result<(), TransportError>;

    /// Reads a control transfer into `data`, returning how many bytes arrived.
    async fn control_in(
        &mut self,
        request: ControlRequest,
        data: &mut [u8],
    ) -> Result<usize, TransportError>;

    /// Reads sample data from the bulk endpoint, returning how many bytes arrived.
    async fn bulk_in(&mut self, data: &mut [u8]) -> Result<usize, TransportError>;
}

/// A recorded transfer, for tests and for logging a real session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Recorded {
    Out {
        value: u16,
        index: u16,
        data: Vec<u8>,
    },
    In {
        value: u16,
        index: u16,
        len: usize,
    },
    Bulk {
        len: usize,
    },
}

/// Transport that records every transfer and replays canned responses.
///
/// The driver's initialisation is a long, ordered sequence where a single misplaced write
/// leaves the device in a state that fails much later and somewhere else. Recording lets a
/// test assert on the exact sequence, which turns that class of bug into a diff.
#[derive(Debug, Default)]
pub struct MockTransport {
    pub log: Vec<Recorded>,
    /// Responses handed back to `control_in`, in order. Once exhausted, reads return
    /// zeroes, which is what an absent device looks like.
    pub responses: std::collections::VecDeque<Vec<u8>>,
    /// Set to make every subsequent transfer fail, to exercise error paths.
    pub fail_after: Option<usize>,
}

impl MockTransport {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queues a response for a later `control_in`.
    pub fn push_response(&mut self, data: impl Into<Vec<u8>>) {
        self.responses.push_back(data.into());
    }

    /// Every write, as `(value, index, data)`.
    pub fn writes(&self) -> Vec<(u16, u16, Vec<u8>)> {
        self.log
            .iter()
            .filter_map(|r| match r {
                Recorded::Out { value, index, data } => Some((*value, *index, data.clone())),
                _ => None,
            })
            .collect()
    }

    /// Whether a write with exactly these arguments was recorded.
    pub fn wrote(&self, value: u16, index: u16, data: &[u8]) -> bool {
        self.log.iter().any(|r| {
            matches!(r, Recorded::Out { value: v, index: i, data: d }
                if *v == value && *i == index && d.as_slice() == data)
        })
    }

