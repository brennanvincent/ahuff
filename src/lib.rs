// This code is inspired by the LZCOMP sample code published
// at https://www.w3.org/Submission/MTX/SUBM-MTX-20080305. Copyright © 2008 Monotype Imaging.
//
// The aforementioned LZCOMP code is made available under the W3C Document License.
//
// All further contributions (including the port to Rust)
// are Copyright © 2022 Brennan Vincent, and made available under the Apache License, Version 2.0.

//! An adaptive Huffman encoder, for use in compression schemes.
//! 
//! To encode data, instantiate a [`Codec`] object, and repeatedly call
//! [`Codec::write_and_update`]. To decode data, instantiate a [`Codec`] object,
//! and repeatedly call [`Codec::read_and_update`].
//!
//! State is shared between the encoder and decoder *implicitly*; that is,
//! it is constructed from the sequence of encoded or decoded symbols, rather
//! than being communicated in the data. Thus, exactly the same data must be decoded on the
//! decode side as was encoded on the encode side, or the decoder will go off the rails
//! and produce nonsense. Furthermore, each side must be constructed with the same parameters,
//! and the same major version of this library must be used.
//!
//! The encoder does not emit any special symbol for EOF. Thus, either the application must
//! designate its own EOF symbol
//! (emitting it on the encoder, and stopping decoding when it is read),
//! or it must somehow communicate the length of the input file out of band
//! and only read that many symbols.

use endio_bit::{BEBitReader, BEBitWriter};

use std::io::{Read, Write};

#[derive(Debug, Clone)]
enum Node {
    /// An internal node, which
    /// must have exactly two children.
    Internal {
        /// Index of the left child.
        left: usize,
        /// Index of the right child.
        right: usize,
        /// Index of the parent, if one exists
        /// (otherwise, we are the root).
        up: Option<usize>,
        /// The summed weights of the subtree rooted here
        weight: u64,
    },
    /// A leaf node, corresponding to one of the symbols.
    Leaf {
        /// The parent node (must exist, as leaves can't be the root).
        up: usize,
        /// The weight of this node
        /// (number of occurrences since rebalance + (weight-before-rebalance / 2))
        weight: u64,
        /// The symbol whose frequency is tracked by this node.
        symbol: usize,
    },
}

#[cfg(test)]
mod tests {
    use std::io::{BufReader, BufWriter};

    use endio_bit::{BEBitReader, BEBitWriter};

    use crate::Codec;

    #[test]
    fn test_rt() {
        let input = "* The rain in Spain stays mainly in the plain.
* Peter piper picked a peck of pickled peppers.
* Four score and seven years ago, our fathers brought forth, upon this continent, a new nation...
* Colorless green ideas sleep furiously.
* Nön-åščīî text should work too"
            .as_bytes();
        let writer = Vec::<u8>::new();
        let writer = BufWriter::new(writer);
        let mut writer = BEBitWriter::new(writer);

        let mut huff = Codec::new(256);

        for &b in input.iter() {
            huff.write_and_update(b as usize, &mut writer).unwrap();
        }
        let result = writer.into_inner().unwrap().into_inner().unwrap();
        // std::fs::write("/tmp/out.huff", &result).unwrap();

        let ratio = input.len() as f64 / result.len() as f64;

        println!(
            "Input len: {}
Result len: {}
Compression ratio: {ratio}",
            input.len(),
            result.len()
        );

        let mut huff = Codec::new(256);
        let reader = BufReader::new(&*result);
        let mut reader = BEBitReader::new(reader);

        let mut rt_result = Vec::new();
        for _ in 0..input.len() {
            let sym = huff.read_and_update(&mut reader).unwrap();
            assert!(sym < 256);
            rt_result.push(sym as u8);
        }
        assert!(rt_result == input);
    }
}

impl Node {
    fn weight(&self) -> u64 {
        match self {
            Node::Internal { weight, .. } => *weight,
            Node::Leaf { weight, .. } => *weight,
        }
    }
    fn up(&self) -> Option<usize> {
        match self {
            Node::Internal { up, .. } => up.clone(),
            Node::Leaf { up, .. } => Some(*up),
        }
    }
    fn incr_weight(&mut self) {
        match self {
            Node::Internal { weight, .. } => *weight += 1,
            Node::Leaf { weight, .. } => *weight += 1,
        }
    }
}

/// The core struct used to encode and decode adaptive Huffman
/// data. Exactly the same series of symbols must be decoded by the decoder
/// as were encoded by the encoder, and the tree must
/// be instantiated with the same parameters on both sides;
/// because the state is updated in a deterministic
/// way, this ensures that the decoder _implicitly_ knows the encoder's state at any point,
/// without having to physically communicate an encoding tree or other information.
#[derive(Debug, Clone)]
pub struct Codec {
    /// The number of symbols in the alphabet.
    n_symbols: usize,
    /// Arena for the binary tree of frequency data.
    /// A few invariants hold:
    /// * `tree.len() == n_symbols * 2`
    /// * `tree[0].is_none()`
    /// * `tree[n].is_some()` for all 1 <= n < tree.len().
    /// * `tree[1]` is the root.
    ///
    /// The reason for the dummy value at tree[0] is that it simplifies some
    /// of the init code to assume that the first half of the array is all
    /// internal nodes and the second half is all leaves.
    tree: Vec<Option<Node>>,
    /// The index of each symbol's node in the tree. Initially `n_symbols + i`, but things will move around.
    symbol_index: Vec<usize>,
}

impl Codec {
    /// Create a new `codec` for an alphabet with the specified number
    /// of symbols, which must be at least two.
    // TODO - support rebalancing
    pub fn new(n_symbols: usize) -> Self {
        assert!(n_symbols > 2);
        let mut tree = vec![None; n_symbols * 2];

        for i in 1..n_symbols {
            tree[i] = Some(Node::Internal {
                left: 2 * i,
                right: 2 * i + 1,
                up: if i == Self::ROOT { None } else { Some(i / 2) },
                weight: 0,
            });
        }
        for i in n_symbols..(2 * n_symbols) {
            tree[i] = Some(Node::Leaf {
                up: i / 2,
                weight: 1,
                symbol: (i - n_symbols),
            });
        }

        fn init_weight(tree: &mut [Option<Node>], i: usize) -> u64 {
            let new_weight = match tree[i].as_ref().unwrap() {
                Node::Internal { left, right, .. } => {
                    let (left, right) = (*left, *right);
                    init_weight(tree, left) + init_weight(tree, right)
                }
                Node::Leaf { weight, .. } => *weight,
            };
            if let Node::Internal { weight, .. } = tree[i].as_mut().unwrap() {
                *weight = new_weight
            }
            new_weight
        }

        init_weight(&mut tree, Self::ROOT);

        let symbol_index = (n_symbols..2 * n_symbols).collect();
        Self {
            n_symbols,
            tree,
            symbol_index,
        }
    }

    /// Emit the specified symbol to the given bitstream, updating the internal state.
    pub fn write_and_update<W>(
        &mut self,
        symbol: usize,
        writer: &mut BEBitWriter<W>,
    ) -> Result<(), std::io::Error>
    where
        W: Write,
    {
        let mut stack = smallvec::SmallVec::<[bool; 50]>::new();
        let mut a = self.symbol_index[symbol];
        let orig_a = a;
        let mut node = self.tree[a].as_ref().unwrap();
        while let Some(up) = node.up() {
            let next = self.tree[up].as_ref().unwrap();
            let (next_left, next_right) = match next {
                &Node::Internal { left, right, .. } => (left, right),
                Node::Leaf { .. } => unreachable!(),
            };
            assert!(next_left == a || next_right == a);
            stack.push(next_right == a);
            a = up;
            node = next;
        }
        for &bit in stack.iter().rev() {
            writer.write_bit(bit)?;
        }
        self.update_at(orig_a);
        Ok(())
    }

    /// Read a symbol from the given bitstream, updating the internal state.
    /// Returns the symbol that was read.
    pub fn read_and_update<R>(
        &mut self,
        reader: &mut BEBitReader<R>,
    ) -> Result<usize, std::io::Error>
    where
        R: Read,
    {
        let mut a = Self::ROOT;
        while let &Node::Internal { left, right, .. } = self.tree[a].as_ref().unwrap() {
            let next = reader.read_bit()?;
            a = if next { right } else { left };
        }
        let symbol = match self.tree[a].as_ref().unwrap() {
            Node::Leaf { symbol, .. } => *symbol,
            Node::Internal { .. } => unreachable!(),
        };
        assert!(self.symbol_index[symbol] == a);
        self.update_at(a);
        Ok(symbol)
    }

    /// Update the internal state assuming the given symbol was read.
    ///
    /// Normally, [`read_and_update`] should be used instead. This function exists
    /// to support use cases where some symbols are communicated through some side channel
    /// other than the main shared bitstream.
    /// (For example, the MTX format assumes non-zero starting counts for various special symbols)
    pub fn update(&mut self, symbol: usize) {
        let a = self.symbol_index[symbol];
        self.update_at(a);
    }
}

// private
impl Codec {
    const ROOT: usize = 1;

    fn update_at(&mut self, mut a: usize) {
        assert!(0 < a && a < 2 * self.n_symbols);
        while a != Self::ROOT {
            // Find the leftmost node whose weight
            // is equal to a's.
            let leftmost_equiv = {
                assert!(a > 1);
                let mut b = a - 1;
                // It's not possible for the root to have the same
                // weight as `a`, because we have asserted that `a` is not the
                // root, and the root has greater weight than any other node,
                // because there are at least two nodes (and the nodes all have weight >= 1).
                //
                // NB: We will have to change this logic if nodes can have zero weight!
                //
                // Thus we will never run left past the root node.
                while self.tree[b]
                    .as_ref()
                    .expect("b is not the dummy node")
                    .weight()
                    <= self.tree[a]
                        .as_ref()
                        .expect("a is not the dummy node")
                        .weight()
                {
                    assert!(
                        self.tree[b].as_ref().unwrap().weight()
                            == self.tree[a].as_ref().unwrap().weight()
                    );
                    b -= 1;
                    assert!(b > 0);
                }
                assert!(b < a);
                b + 1
            };
            // As above: the root cannot be equivalent to `a`.
            assert!(Self::ROOT < leftmost_equiv && leftmost_equiv <= a);
            // Actually, a stronger statement is true: neither `a`
            // nor `b` can strictly descend from the other. Otherwise, it would not
            // be possible for them to have the same value.
            self.swap_nodes(a, leftmost_equiv);
            a = leftmost_equiv;
            self.tree[a].as_mut().unwrap().incr_weight();
            self.assert_weights_add_at(a);
            a = self.tree[a]
                .as_ref()
                .unwrap()
                .up()
                .expect("a is known not to be the root");
        }
        assert!(a == Self::ROOT);
        self.tree[a].as_mut().unwrap().incr_weight();
        self.assert_weights_add_at(a);
    }
    fn assert_weights_add_at(&self, a: usize) {
        if let Node::Internal {
            left,
            right,
            weight,
            ..
        } = self.tree[a].as_ref().unwrap()
        {
            assert!(
                self.tree[*left].as_ref().unwrap().weight()
                    + self.tree[*right].as_ref().unwrap().weight()
                    == *weight
            );
        }
    }
    fn is_a_ancestor_of_b(&self, a: usize, mut b: usize) -> bool {
        if a == b {
            return true;
        }
        while let Some(up) = self.tree[b].as_ref().unwrap().up() {
            if up == a {
                return true;
            }
            b = up;
        }
        false
    }
    /// Precondition: neither `a` nor `b`
    /// is a strict ancestor of the other
    /// (corrolary: neither is the root)
    fn swap_nodes(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        debug_assert!(!(self.is_a_ancestor_of_b(a, b) || self.is_a_ancestor_of_b(b, a)));
        fn set_up(node: &mut Node, new_up: usize) {
            match node {
                Node::Internal { up, .. } => *up = Some(new_up),
                Node::Leaf { up, .. } => *up = new_up,
            }
        }
        let up_a = self.tree[a]
            .as_ref()
            .unwrap()
            .up()
            .expect("a is not the root");
        let up_b = self.tree[b]
            .as_ref()
            .unwrap()
            .up()
            .expect("b is not the root");

        // TODO more asserts throughout this function
        self.tree.swap(a, b);
        set_up(self.tree[a].as_mut().unwrap(), up_a);
        set_up(self.tree[b].as_mut().unwrap(), up_b);

        match self.tree[a].as_ref().unwrap() {
            &Node::Internal { left, right, .. } => {
                set_up(self.tree[left].as_mut().unwrap(), a);
                set_up(self.tree[right].as_mut().unwrap(), a);
            }
            &Node::Leaf { symbol, .. } => {
                self.symbol_index[symbol] = a;
            }
        }
        match self.tree[b].as_ref().unwrap() {
            &Node::Internal { left, right, .. } => {
                set_up(self.tree[left].as_mut().unwrap(), b);
                set_up(self.tree[right].as_mut().unwrap(), b);
            }
            &Node::Leaf { symbol, .. } => {
                self.symbol_index[symbol] = b;
            }
        }
    }
}
