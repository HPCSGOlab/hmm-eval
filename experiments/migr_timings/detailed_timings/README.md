custom driver within ~/hmm-eval/drivers/x86_64-560.35.05/vanilla/kernel-open/ must be loaded for this to work as we use the dmesg timings


any size over 8192 for SGEMM overflows the dmesg and therefore we cannot get accurate timings.


currently only automatted for SGEMM
