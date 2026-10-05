# Arithmetic findings in the current tiled ResNet-18 generator

These behaviors are reproduced by the implementation reference and by the
generated C++ operators compiled with actual Vitis 2024.1 fixed-point headers.
They are deliberately preserved; fixing the reduction algorithms is a separate
change from building the validation workflow.

## Incoming chunks do not account for the shared exponent

Convolution and FC divide their accumulator mantissas by two when the guard is
exceeded, then add later raw products without dividing those contributions by
the accumulated exponent.

The focused FC case has 512 inputs equal to 1. The first 128 weights equal 3,
the next 128 equal -1, and the remaining weights are zero. Output shift is 5.

* Mathematical sum: `384 - 128 = 256`; scaled output: `256 / 32 = 8`.
* Current implementation: first chunk rescales `384 -> 192`, exponent 1;
  second chunk produces `192 - 128 = 64`; commit produces `64 / 16 = 4`.

No accumulator saturation is needed to expose this defect. The final output
also stays in range, so storage saturation does not conceal it.

## Guard checks occur after potentially saturating reductions

Each `sum += product` can saturate before the chunk-level guard is checked.
Once it saturates, subsequent renormalization cannot recover the lost value.

The focused FC case uses 128 products equal to 8, followed by zeros, and output
shift 7. The intended output is `1024 / 128 = 8`. The implementation saturates
near 512 before renormalization and ultimately stores 4.

## Global average pooling can saturate its sum

The current pooling code finishes all 49 additions before checking its shared
exponent, and restores the exponent in `acc_t` before dividing by 49.
For a constant 7x7 map equal to 12, the sum should be 588 and the average 12.
The accumulator saturates near 512, so the stored average is approximately
10.449. The regression also covers the negative case and lower constants.

## Expression widths exceed the declared accumulator width

Although stored accumulators are `ap_fixed<32,10,AP_RND,AP_SAT>`, the vendor's
operator rules produce wider temporary types. Addition yields 33 bits and
multiplication yields 64 bits for two accumulator operands. In BN, the square
root argument/result is `ap_fixed<33,11>` and normalization division produces
`ap_fixed<56,34>`. This matters to numerical emulation and to interpreting any
claimed cap on all internal expression widths. No width-policy changes are made
by this verification implementation.

## Compilation-only corrections

Array parameters decay to pointers, preventing deduction of their leading
dimension in several helper templates and the head operators. Their call sites
now provide explicit dimensions. The absolute-value helper now uses branches
instead of a conditional expression with mutually convertible fixed-point
types. These changes permit compilation without changing the intended casts,
loop ordering, arithmetic, or public memory interface.
