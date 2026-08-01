# Training Log (Historical)

This log covered the W_pred era (D=3584, L=32, 221M). Current architecture: D=4096/896, G=32/8,
~255M/~17.6M (default, SwiGLU + BottleneckBind shift), GroupedCognitiveMirror. See
`ANALYSIS_REPORT.md` for training notes, `README.md` for architecture. Note: pre-Aug-2026
metrics were produced on a corrupted corpus (~92% U+FFFD) and are treated as artifacts.
