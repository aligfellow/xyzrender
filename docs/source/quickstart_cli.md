# CLI Quickstart

One command is all you need:

```bash
xyzrender caffeine.xyz
```

This writes `caffeine.svg` in the current directory, auto-oriented with depth cueing and bond orders.

```{image} ../../examples/images/caffeine_default.svg
:width: 400
:alt: Caffeine default render
```

Specify output path and format with `-o` (extension controls format):

```bash
xyzrender caffeine.xyz -o render.svg
xyzrender caffeine.xyz -o render.png
xyzrender caffeine.xyz -o render.pdf
```

From QM output (ORCA, Gaussian, Q-Chem — auto-detected from content):

```bash
xyzrender calc.out
```

From stdin:

```bash
cat caffeine.xyz | xyzrender
```

See [Input formats](formats.md), [Configuration](configuration.md), [Orientation](orientation.md), and the full [CLI reference](cli_reference.md) for everything else.
