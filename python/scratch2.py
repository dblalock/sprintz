#!/usr/bin/env python

# import collections
# import itertools
# import os
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sb

# from .scratch1 import nbits_cost

from .compress import nbits_cost, zigzag_encode   # TODO rm this line
from . import compress

# from .datasets import ucr
# from . import datasets as ds
# from .utils import files
# from .utils import sliding_window as window


# class SprintzEncoder(object):

#     def __init__(self, history_len, nn_step, hash_algo='perm'):
#         self.nn_step = nn_step
#         self.hash_algo = hash_algo
#         # self.history = np.zeros(history_len
#         self.history_len = history_len  # TODO have a circ buff

#     def encode(self, all_samples):  # TODO enc one block at a time
#         """all_samples: 1D array"""
#         diffs =


# def _pseudocounts(nbits):
#     assert nbits == 8

#     vals = np.arange(-128, 128)
#     costs = nbits_cost(vals)
#     pseudocounts = nbits - costs

#     assert np.min(pseudocounts) == 0
#     return pseudocounts


# Symbol = collections.namedtuple('Symbol', 'val count'.split())

# ================================================================ SortTransform

class Symbol(object):
    __slots__ = 'val count'.split()

    def __init__(self, val, count):
        self.val = val
        self.count = count


def min_max_vals_for_nbits(nbits, signed=True):
    if signed:
        minval = -(1 << (nbits - 1))  # for 8b, -128
        maxval = -minval - 1  # for 8b, 127
        return minval, maxval
    return 0, (1 << nbits) - 1


def _initial_symbols_positions(nbits, signed=True):
    minval, maxval = min_max_vals_for_nbits(nbits, signed=signed)

    if signed:
        # vals = 0, -1, 1, -2, 2, etc
        vals = [0]
        for i in range(1, maxval + 1):
            vals += [-i, i]
        vals += [minval]
        vals = np.array(vals)

        idxs = np.array([int(np.where(vals == i)[0]) for i in np.arange(minval, maxval + 1)])
    else:
        # vals = 0, 1, 2, 3, etc
        idxs = np.arange(minval, maxval + 1)  # 0, 2^nbits - 1
        vals = np.copy(idxs)

    # pseudocount of nbits - (nbits needed to express the value)
    pseudocounts = [nbits - nbits_cost(val, signed=signed) for val in vals]
    symbols = [Symbol(val=val, count=count) for val, count in zip(vals, pseudocounts)]

    # print "using vals: ", vals
    # print "using idxs: ", idxs
    # ordered_vals = np.arange(minval, maxval + 1)
    # for i, val in enumerate(ordered_vals):
    #     idx = idxs[val + minval]
    #     assert vals[idx] == val

    return symbols, idxs.astype(np.int32)


class SortTransformer(object):
    __slots__ = 'symbols positions minval maxval'.split()

    def __init__(self, nbits=8, signed=True):
        self.symbols, self.positions = _initial_symbols_positions(nbits, signed=signed)
        # self.minval = -(1 << (nbits - 1))  # for 8b, -128
        self.minval, self.maxval = min_max_vals_for_nbits(nbits, signed=signed)

        # check that everything is sorted # TODO rm
        counts = self.counts()
        # print "initial counts: ", counts
        # print "minval, positions", self.minval, self.positions
        diffs = np.diff(counts)
        assert np.all(diffs <= 0) or False

    # we maintain two lists acting as maps:
    #   symbols: idx -> val, count
    #   positions: val -> idx
    def feed_val(self, val, move_limit=2):
        symbols = self.symbols  # abbreviate since written a lot here
        # idx = self.positions[val + self.minval]
        idx = self.positions[val - self.minval]
        symbol = Symbol(val=symbols[idx].val, count=symbols[idx].count)
        symbol.count += 1

        # assert self.minval <= val <= self.maxval

        # return # TODO rm

        # assert self.minval == 0

        it = 0
        move_limit = move_limit if move_limit > 0 else len(self.positions)
        # wow; making this >= instead of > helps an *enormous* amount
        # EDIT: that was before we were decrementing idx, so who knows what
        # that was doing...
        while symbol.count >= symbols[idx - 1].count and idx > 0 and it < move_limit:
            other_val = symbols[idx - 1].val

            # assert 0 <= other_val < 256
            # assert 0 <= val < 256

            # self.positions[other_val + self.minval] += 1
            # self.positions[val + self.minval] -= 1
            self.positions[other_val - self.minval] += 1
            self.positions[val - self.minval] -= 1

            # swap the two symbols; have to set the fields explicitly
            # for subtle python reasons; ie, obvious way below doesn't work:
            #
            # symbols[idx], symbols[idx - 1] = symbols[idx - 1], symbol
            #
            symbols[idx].val = symbols[idx - 1].val
            symbols[idx].count = symbols[idx - 1].count
            symbols[idx - 1].val = symbol.val
            symbols[idx - 1].count = symbol.count

            idx -= 1
            it += 1

        # # check that everything is sorted # TODO rm
        # counts = self.counts()
        # # print "updated vals: ", self.sorted_vals()
        # # print "updated positions: ", self.positions
        # # print "updated counts: ", counts
        # diffs = np.diff(counts)
        # assert np.all(diffs <= 0)

    def position_of_val(self, val):
        # return self.positions[val + self.minval]
        return self.positions[val - self.minval]

    def counts(self):
        return np.array([s.count for s in self.symbols])

    def sorted_vals(self):
        return np.array([s.val for s in self.symbols])


def sort_transform(blocks, nbits=16, prefix_nbits=-1,
                   # zigzag=True, sort_remainders=True):
                   zigzag=False,
                   # zigzag=True,
                   # sort_remainders=True):
                   sort_remainders=False):
    """
    >>> x = [3, -4, 0, 1, -2]
    >>> sort_transform(x, nbits=4)
    array([6, 7, 0, 2, 3])
    """

    # prefix_nbits = 9  # TODO rm
    # return (np.abs(blocks) << 1) - (blocks > 0).astype(np.int32) # TODO rm

    if sort_remainders:
        assert 0 < prefix_nbits < nbits  # no remainders without suffix bits
    if prefix_nbits is None or prefix_nbits < 1:
        prefix_nbits = nbits

    assert prefix_nbits <= nbits
    suffix_nbits = nbits - prefix_nbits
    max_trailing_val = max((1 << suffix_nbits) - 1, 0)

    blocks = np.asarray(blocks).astype(np.int32)
    out = np.empty(blocks.size, dtype=blocks.dtype)

    if zigzag:
        blocks = (np.abs(blocks) << 1) - (blocks > 0).astype(np.int32)

    signed = not zigzag
    # signed = True # TODO rm
    transformer_high = SortTransformer(nbits=prefix_nbits, signed=signed)
    if suffix_nbits and sort_remainders:
        transformer_low = SortTransformer(nbits=suffix_nbits, signed=signed)

    for i, val in enumerate(blocks.ravel()):
        val_low = np.bitwise_and(val, max_trailing_val)
        val_high = val >> suffix_nbits
        try:
            if suffix_nbits and sort_remainders:
                pos_high = transformer_high.position_of_val(val_high)
                pos_low = transformer_low.position_of_val(val_low)

                out[i] = (pos_high << suffix_nbits) + pos_low

                transformer_high.feed_val(val_high)
                transformer_low.feed_val(val_low)
            else:
                pos_high = transformer_high.position_of_val(val_high)
                out[i] = (pos_high << suffix_nbits) + val_low
                transformer_high.feed_val(val_high)

        except IndexError:  # happens when deltas overflow
            out[i] = (1 << (nbits - 1)) - 1

    out = out.reshape(blocks.shape)
    # if zigzag:
    return compress.zigzag_decode(out)  # always undo nonneg idx
    # return out


def sign_extend(value, nbits):
    value = np.asarray(value)
    sign_bit = 1 << (nbits - 1)
    return np.bitwise_xor(value, sign_bit) - sign_bit


def prefix_lut_transform(blocks, prefix_nbits=4):
    assert prefix_nbits > 0

    # let bmax = max nbits used by any block
    # let counts = counts of each length prefix_nbits prefix, starting at bmax
    # sort prefixes in descending order of frequency
    # for each block
    #   replace its prefix (again starting at bmax) with its sorted idx
    #
    # to reconstruct, we would always shift stuff so that bmax was msb, before
    # right shifting to sign extend

    costs = nbits_cost(blocks)
    block_costs = np.max(costs, axis=1)
    bmax = np.max(block_costs)
    # ^ could just take max cost total; in practice, OR together the zigzag
    # encoding of all the samples we ever see, then count leading 0s

    bmin = max(0, bmax - prefix_nbits)
    if bmin == bmax:
        return blocks

    blocks_flat = blocks.astype(np.int32).ravel()
    prefixes = blocks_flat >> bmin

    mask = (1 << prefix_nbits) - 1
    prefixes = np.bitwise_and(prefixes, mask)
    counts = np.bincount(prefixes)
    print "prefix counts: ", counts

    # this unpacks into prefixes and low bits, then reconstructs input
    # ret = sign_extend(prefixes << bmin, bmax) # TODO rm
    # low_bits = np.bitwise_and(blocks_flat, (1 << bmin) - 1)
    # ret |= low_bits
    # return ret.reshape(blocks.shape)

    idxs = np.argsort(counts)
    decode_lut = idxs[::-1]  # most common prefixes in decreasing order
    encode_lut = np.zeros_like(decode_lut)
    encode_lut[decode_lut] = np.arange(len(encode_lut))
    # for i, idx in enumerate(decode_lut):
    #     encode_lut[idx] = i

    prefixes_enc = encode_lut[prefixes.ravel()]
    print "encoded prefix counts: ", np.bincount(prefixes_enc)

    assert prefix_nbits == 4  # XXX
    convert_to_signed = [0, 15, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9, 7, 8]
    convert_to_signed = np.array(convert_to_signed)
    prefixes_enc = convert_to_signed[prefixes_enc]

    # print "signed encoded prefix counts: ", np.bincount(prefixes_enc)

    low_bits = np.bitwise_and(blocks_flat, (1 << bmin) - 1)
    # hack to deal with prefix swapping signs
    actual_prefix_nbits = bmax - bmin
    assert actual_prefix_nbits == 4
    msbs_orig = prefixes >> (actual_prefix_nbits - 1)
    msbs_new = prefixes_enc >> (actual_prefix_nbits - 1)
    xor_indicators = (msbs_new != msbs_orig).astype(np.int32)
    xor_masks = (xor_indicators << bmin) - xor_indicators
    low_bits = np.bitwise_xor(low_bits, xor_masks)
    # low_bits = 0 # TODO rm
    # low_bits = np.bitwise_and(blocks.astype(np.int32).ravel(), (1 << bmin) - 1) # TODO rm

    # print "xor indicators[50:55]", xor_indicators[50:70]
    # print "xor masks[50:55]", xor_masks[50:70]
    # print "number of sign changes: ", np.sum(xor_indicators)

    ret = np.bitwise_or(low_bits, prefixes_enc << bmin)
    return sign_extend(ret, bmax).reshape(blocks.shape)

    # sneaky sign extension
    # sign_bit = 1 << bmax
    # return ret - np.bitwise_and(ret, sign_bit)


# ================================================================ MixFix

POWER_LAW_CODES_2 = np.array([
    '0',
    '1',
    ])
POWER_LAW_CODELENGTHS_2 = np.array([1, 1], dtype=np.int32)

POWER_LAW_CODES_4 = np.array([
    '0',
    '10',
    '110',
    '111'
    ])
POWER_LAW_CODELENGTHS_4 = np.array([len(s) for s in POWER_LAW_CODES_4],
                                   dtype=np.int32)

# POWER_LAW_CODES_8 = np.array([
#     '0',
#     '10',
#     '1100',
#     '1101',
#     '11100',
#     '11101',
#     '11110',
#     '11111'
#     ])
POWER_LAW_CODES_8 = np.array([
    '00',
    '01',
    '100',
    '101',
    '1100',
    '1101',
    '1110',
    '1111'
    ])
POWER_LAW_CODELENGTHS_8 = np.array([len(s) for s in POWER_LAW_CODES_8],
                                   dtype=np.int32)

# POWER_LAW_CODES_16 = np.array([
#     '0',
#     '100',
#     '1010',
#     '1011',
#     '11000',
#     '11001',
#     '11010',
#     '11011',
#     '111000',
#     '111001',
#     '111010',
#     '111011',
#     '111100',
#     '111101',
#     '111110',
#     '111111'
#     ])
# POWER_LAW_CODES_16 = np.array([
#     '000',
#     '001',
#     '0100',
#     '0101',
#     '0110',
#     '0111',
#     '1000',
#     '1001',
#     '1010',
#     '1011',
#     '1100',
#     '1101',
#     '11100',
#     '11101',
#     '11110',
#     '11111'
#     ])
# POWER_LAW_CODES_16 = np.array([
#     '000',
#     '001',
#     '010',
#     '011',
#     '1000',
#     '1001',
#     '1010',
#     '1011',
#     '11000',
#     '11001',
#     '11010',
#     '11011',
#     '11100',
#     '11101',
#     '11110',
#     '11111'
#     ])
POWER_LAW_CODES_16 = np.array([  # cheap 0s, but otherwise almost unif
    '00',
    '0100',
    '0101',
    '0110',
    '0110',
    '1001',
    '1000',
    '1011',
    '1010',
    '11000',
    '11001',
    '11010',
    '11011',
    '11100',
    '11101',
    '11110'
    ])
POWER_LAW_CODELENGTHS_16 = np.array([len(s) for s in POWER_LAW_CODES_16],
                                    dtype=np.int32)

ALL_POWER_LAW_CODES = [POWER_LAW_CODES_2, POWER_LAW_CODES_4,
                       POWER_LAW_CODES_8, POWER_LAW_CODES_16]
ALL_POWER_LAW_CODELENGTHS = [POWER_LAW_CODELENGTHS_2, POWER_LAW_CODELENGTHS_4,
                             POWER_LAW_CODELENGTHS_8, POWER_LAW_CODELENGTHS_16]

ENCODING_RICE = 'rice'
ENCODING_POWER_LAW = 'pwr'
ENCODING_UNIF = 'unif'


def _mixfix_pick_encoding(block, bits, b_suffix_min, b_suffix_max):
    # min_val, max_val = np.min(block), np.max(block)
    # print "block: ", block

    min_nbits, max_nbits = np.min(bits), np.max(bits)
    block = block.astype(np.int32)
    block_sz = len(block)
    block_sz_log2 = int(np.log2(block_sz))
    assert 1 << block_sz_log2 == block_sz  # assert block size is power of 2

    # print "block: ", block
    # print "min nbits, max nbits, (gap):\t{}, {}, ({})".format(
    #     min_nbits, max_nbits, max_nbits - min_nbits)
    # print "min nbits, max nbits, suffix_min, suffix_max:\t{}, {}, {}, {}".format(
    #     min_nbits, max_nbits, b_suffix_min, b_suffix_max)

    cost_unif = block_sz * max_nbits
    uniform_retval = ENCODING_UNIF, block_sz * max_nbits, max_nbits, None, None

    powerlaw_codebook_nbits = 4

    # invariants we enforce (by just using a uniform distro if they're false):
    #   max_nbits >= min_nbits + 2      # varlen prefixes can help
    #   max_nbits >= b_suffix_min + 2   # varlen prefixes can help
    #   max_nbits <= b_suffix_max + 4   # prefixes won't overflow codebook
    #
    use_unif = max_nbits < (min_nbits + 2)
    use_unif = use_unif or (max_nbits < (b_suffix_min + 2))
    use_unif = use_unif or (max_nbits > (b_suffix_max + powerlaw_codebook_nbits))
    if use_unif:
        return uniform_retval

    # print "not returning uniform distro!"

    # if :
    #     return uniform_retval
    # if :
    #     return uniform_retval  # min from delta (almost) exceeds max
    # if :
    #     return uniform_retval  # max from delta not large enough
        # if max_nbits == 0:
            # cost = block_sz
            # b_suffix = 0
            # return ENCODING_RICE, cost, b_suffix, None, None
        # else:
        # cost = block_sz * max_nbits
        # b_suffix = max(0, max_nbits - 1)
        # return ENCODING_POWER_LAW, cost, b_suffix, POWER_LAW_CODES_2, POWER_LAW_CODELENGTHS_2

    # number of bits for suffixes, which are fixed-size;
    # rice_min_nbits = max_nbits - 2
    # pwr_min_nbits = max_nbits - 4

    # handle case of min and max being equal
    # b_suffix_rice = min(b_suffix_rice, min_nbits - 1)
    # b_suffix_pwr = min(b_suffix_pwr, min_nbits - 1)

    # rice_possible = rice_min_nbits >= b_suffix_max
    # power_possible = pwr_min_nbits >= b_suffix_max
    # # power_possible = power_possible and (b_suffix_min <= max_nbits)
    # # power_possible = False # TODO rm
    # # rice_possible = False # TODO rm
    # if not (rice_possible or power_possible) or b_suffix_min >= (max_nbits + 1):
    #     return uniform_retval
    #     # return ENCODING_UNIF, block_sz * max_nbits, max_nbits, None, None
    #     # raise ValueError(
    #         # "Block requires b_suffix of at least '{}', but max is {}".format(
    #             # b_suffix_pwr, b_suffix_max))

    # clip suffix length based on constraints; note that making it larger
    # than optimal is always okay
    lb_nbits = max(min_nbits, b_suffix_min)
    b_suffix_rice = max(lb_nbits, max_nbits - 2)
    b_suffix_pwr = max(lb_nbits, max_nbits - 4)

    # cost of rice coding
    prefix_vals = block >> b_suffix_rice  # vals in {0, 1, ..., 7}
    # print "block: ", block
    # print "b_suffix_rice: ", b_suffix_rice
    # print "prefix_vals: ", prefix_vals
    assert 0 <= np.max(prefix_vals) <= 7
    cost_rice = int(np.sum(prefix_vals)) + block_sz  # costs of {1,...,8}

    # TODO rm
    # return ENCODING_RICE, cost_rice, b_suffix_rice, None, None

    # cost of encoding with power law codes above
    gap = max_nbits - b_suffix_pwr
    # idx = max(0, gap - 1)  # b_suffix_min can be >= max_nbits
    idx = gap - 1
    codebook = ALL_POWER_LAW_CODES[idx]
    codelengths = ALL_POWER_LAW_CODELENGTHS[idx]
    prefixes = block >> b_suffix_pwr
    prefix_costs = codelengths[prefixes]
    cost_power = np.sum(prefix_costs) + (b_suffix_pwr << block_sz_log2)

    # print "min nbits, max nbits, suffix_min, suffix_max, (pwr law gap) <b_suffix>:" \
    #     "\t{}, {}, {}, {}, ({}) <{}>".format(
    #         min_nbits, max_nbits, b_suffix_min, b_suffix_max, gap, b_suffix_pwr)
    # print "block: ", block
    # print "prefixes: ", prefixes
    # print "prefix costs: ", prefix_costs
    # print "costs: ", prefix_costs + b_suffix_pwr

    # if cost_rice < cost_power:

    if False:  # TODO rm
        return ENCODING_RICE, cost_rice, b_suffix_rice, None, None
    else:
        # print "min nbits, max nbits, (pwr law gap) <b_suffix>:\t" \
        #     "{}, {}, ({}) <{}>".format(min_nbits, max_nbits, gap, b_suffix_pwr)
        # bitsave = cost_unif - cost_power
        # print "pwr saved {}b".format(bitsave)
        if cost_power < cost_unif:
            return ENCODING_POWER_LAW, cost_power, b_suffix_pwr, codebook, codelengths

        return uniform_retval


def _to_unary(x):
    """
    >>> [_to_unary(i) for i in [0, 1, 2, 3, 4]]
    [0, 2, 6, 14, 30]
    """
    return np.maximum(0, (1 << (x + 1)) - 2)


def _mixfix_encode(block, encoding_name, b_suffix, codebook, codelengths):

    heads = block >> b_suffix
    mask = (1 << b_suffix) - 1
    suffixes = np.bitwise_and(block, mask)

    if encoding_name == ENCODING_RICE:
        prefixes = _to_unary(heads)
        prefix_lengths = (1 << heads)
        codes = (prefixes << b_suffix) + suffixes
    elif encoding_name == ENCODING_POWER_LAW:
        prefixes = codebook[heads]
        prefix_lengths = codelengths[heads]
        codes = None  # avoid str to int conversion cuz just prototyping
    elif encoding_name == ENCODING_UNIF:
        return block, b_suffix
    else:
        raise ValueError("Unrecognized encoding '{}'".format(encoding_name))

    bitlengths = prefix_lengths + b_suffix
    return codes, bitlengths


class MixFixEncoder(object):
    __slots__ = 'nbits b_suffix max_suffix_delta nbits_mins' \
        ' nbits_maxes gaps'.split(' ')

    def __init__(self, nbits, suffix_delta_nbits=4):
        self.nbits = nbits
        self.b_suffix = nbits - 3
        self.max_suffix_delta = (1 << (suffix_delta_nbits - 1)) - 1
        self.nbits_mins = []
        self.nbits_maxes = []
        self.gaps = []

    def feed_block(self, block):
        # print "feed_block block: ", block
        block = zigzag_encode(block)
        # print "feed_block block: ", block
        bits = nbits_cost(block, signed=False)

        min_nbits, max_nbits = np.min(bits), np.max(bits)
        self.nbits_mins.append(min_nbits)
        self.nbits_maxes.append(max_nbits)
        self.gaps.append(max_nbits - min_nbits)

        # return block, bits # TODO rm after debug

        min_b_suffix = self.b_suffix - self.max_suffix_delta
        max_b_suffix = self.b_suffix + self.max_suffix_delta

        encoding_name, encoding_cost, b_suffix, codebook, codelens = \
            _mixfix_pick_encoding(block, bits, min_b_suffix, max_b_suffix)

        # if encoding_name != ENCODING_UNIF:
        #     print "encoding {} saved {}b".format(
        #         encoding_name, (max_nbits * block.size) - encoding_cost)

        # print "encoding, cost (b), b_suffix (b):\t{}, {}, {}".format(
        #     encoding_name, encoding_cost, b_suffix)

        self.b_suffix = b_suffix
        return _mixfix_encode(block, encoding_name, b_suffix, codebook, codelens)


def mixfix_encode(blocks, nbits):
    encoder = MixFixEncoder(nbits)
    out = np.empty(blocks.shape, dtype=blocks.dtype)
    for i, block in enumerate(blocks):
        out[i], _ = encoder.feed_block(block)
    return out


def mixfix_cost(blocks, nbits):
    encoder = MixFixEncoder(nbits)
    out = np.empty(blocks.shape, dtype=blocks.dtype)
    for i, block in enumerate(blocks):
        _, out[i] = encoder.feed_block(block)

        # print "======"
        # if i == 10:
        #     import sys; sys.exit()

    # # TODO rm
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(3)
    # encoder.gaps[0] = 0  # keep x axis from getting messed up
    # bins = np.arange(np.max(encoder.gaps) + 1)
    # axes[0].set_title('Min nbits in blocks')
    # axes[0].hist(encoder.nbits_mins, normed=True, bins=bins)
    # axes[1].set_title('Max nbits in blocks')
    # axes[1].hist(encoder.nbits_maxes, normed=True, bins=bins, color='brown')
    # axes[2].set_title('nbits gaps in blocks')
    # axes[2].hist(encoder.gaps, normed=True, bins=bins, color='green')
    # fig.tight_layout()

    return out


# ================================================================ fastpfor

def fastpfor_encode(blocks, pfor_block_sz=128):
    pass # TODO


# ================================================================ main

def main():
    x = [3, -4, 0, 1, -2]
    sort_transform(x, nbits=4)

    x = [3, -4, 0, 1, -2]
    sort_transform(x, nbits=4)

    # x = np.array([0, -1, 1, -2, 2]).reshape((1, -1))
    # mixfix_encode(x, nbits=4)

    # yep, looks good
    # print sign_extend([0, 0], 1)  # 0 0
    # print sign_extend([1, 1], 1)  # -1 -1
    # print sign_extend([1, 1], 2)  # 1 1
    # print sign_extend([2, 1], 2)  # -2 1


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    main()
