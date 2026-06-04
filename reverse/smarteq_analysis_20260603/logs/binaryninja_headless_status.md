# Binary Ninja Headless Status

Binary Ninja GUI was opened with:

```bash
open -a "/Applications/Binary Ninja.app" /Library/Audio/Plug-Ins/VST3/smartEQ4.vst3/Contents/MacOS/smartEQ4
```

The GUI process was observed after launch.

A Python API check was attempted using the Binary Ninja app resource paths and
`bn.load()` against the same `smartEQ4` VST3 executable. The process produced
no useful stdout before exceeding the interactive wait budget and was stopped.

The rest of the analysis uses reproducible read-only macOS static tools and is
recorded in the sibling log files.
