.p2align 4,,15
.global hmx_acc
.type hmx_acc, @function
hmx_acc:
{
    activation.hf = mxmem(R0, R1):deep;
    weight.hf = mxmem(R2, R3);
}
{
    jumpr R31;
}
.size hmx_acc, .-hmx_acc
#if __HEXAGON_ARCH__ < 73   /* legacy accumulator store (V69..V79); writes zeros on V81 */
.p2align 4,,15
.global hmx_store_v69
.type hmx_store_v69, @function
hmx_store_v69:
{
    mxmem(R0, R1):after.hf = acc;
}
{
    jumpr R31;
}
.size hmx_store_v69, .-hmx_store_v69
#endif
/* v73+ accumulator store: (out R0, ctrl R1, scale/bias block R2) bias = mxmem2(R2); cvt.hf = acc(#2); mxmem(R0, R1) = cvt.
 * In a v69 build the three instructions are emitted as raw words (encodings taken from a -mv73 assembly), so one skel serves every
 * generation and the store is picked at run time (gemm2.c, hmx_scales_init). */
.p2align 4,,15
.global hmx_store_v73
.type hmx_store_v73, @function
hmx_store_v73:
#if __HEXAGON_ARCH__ >= 73
{
    bias = mxmem2(R2)
}
{
    R3 = #2
}
{
    cvt.hf = acc(R3)
}
{
    mxmem(R0, R1) = cvt
}
#else
    .word 0x9202c3fe
{
    R3 = #2
}
    .word 0xa6e3da10
    .word 0xa6e0c118
#endif
{
    jumpr R31
}
.size hmx_store_v73, .-hmx_store_v73
