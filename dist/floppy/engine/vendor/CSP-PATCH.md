# Static Embind compatibility patch

The served `mujoco.js` is derived from the official
[`@mujoco/mujoco@3.8.0`](https://registry.npmjs.org/@mujoco/mujoco/-/mujoco-3.8.0.tgz)
default single-threaded loader. Its two dynamic JavaScript compilation sites
prevent initialization under the Arena's strict CSP. Allowing WebAssembly
compilation does not allow JavaScript `Function` compilation.

Only two functions are replaced with the synchronous `DYNAMIC_EXECUTION=0`
fallbacks already provided by Emscripten 4.0.10, commit
`b7dc6e5747465580df5984e723b9d1f10d8e804b`:

- [`craftInvokerFunction`](https://github.com/emscripten-core/emscripten/blob/b7dc6e5747465580df5984e723b9d1f10d8e804b/src/lib/libembind.js)
  marshals the original class receiver/arguments and invokes the original WASM
  function, then performs the same destructor and return conversions.
- [`_emval_get_method_caller`](https://github.com/emscripten-core/emscripten/blob/b7dc6e5747465580df5984e723b9d1f10d8e804b/src/lib/libemval.js)
  uses ordinary closures with `Reflect.construct` / `Function.prototype.apply`
  to call an existing function. It does not compile a string.

The existing `createNamedFunction` already uses `Object.defineProperty`; it
does not need a patch. The binary WASM and all other loader code are unchanged.
This is a narrowly patched loader, not a complete Emscripten rebuild.

Original loader SHA256:
`38caa5929112452a03b7921a5d81fc3e6b05619e8d446995128d93a906848575`

Served patched loader SHA256:
`66777739ae30c38c7ead0221a79e05cbcbad31da3eabeea8fb3f4789c80f22cc`

WASM SHA256:
`a91233f2b4d77bdac310e2050a07e3b882da7fade32d94c80c960a9ec5f5ea53`

To reproduce, extract the original loader from the pinned npm tarball and run
`patch -p0 < embind-static-csp.patch` in its directory. The patch is retained
next to this note. The original MuJoCo Apache license remains in `LICENSE`;
the full Emscripten copyright and permissive license are in
`EMSCRIPTEN_LICENSE`. The ordinary engine tests also pass under
`node --disallow-code-generation-from-strings --test test/floppy-engine.test.mjs`.
Browser checks under the intended CSP are recorded separately by the release.
