# M121 CP01 patch review

Initial verdict: `CHANGES REQUIRED`.

The first implementation validated npm lifecycle variables inside JavaScript.
Independent reproduction showed that this is too late:

```text
npm_config_script_shell=/usr/bin/true npm run mac:release
exit 0
```

No wrapper or core builder executed. The initial synthetic test therefore did
not model the real pre-execution boundary. npm also supplied `INIT_CWD` and
`NODE`, which were not included in the original scrub claim.

Required correction: remove `mac:release` from package scripts; use only the
direct Node launcher; reject any inherited npm, `INIT_CWD`, `NODE` or
`NODE_OPTIONS` surface; set trusted PATH; and add a real npm integration test
showing hostile script-shell cannot return success for a missing release script.

This review contacted no backend, audio device, OSC or WING.

Second review also blocked the direct-JavaScript revision: `NODE_OPTIONS` can
execute before JavaScript and delete its own environment evidence. The accepted
correction therefore moves the boundary to `/bin/sh -p`, unsets Node startup
inputs, and executes the inner verifier through `/usr/bin/env -i`. Direct `.mjs`
entry without the clean-environment marker is rejected.
