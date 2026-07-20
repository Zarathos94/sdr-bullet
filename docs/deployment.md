# Deployment

The app has one hard hosting requirement: the page must be **cross-origin isolated**, or
`SharedArrayBuffer` is undefined and the pipeline cannot start. That means two response
headers on the document:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

`require-corp` has a consequence worth understanding before turning it on: every
cross-origin subresource the page loads must itself opt in, by carrying either a
`Cross-Origin-Resource-Policy` header or proper CORS headers. Anything that does not — a font
from a CDN, an analytics script, a third-party iframe — is blocked. So the headers should be
scoped to this app's route rather than applied site-wide, unless the whole site has already
been audited for cross-origin isolation.

The Vite dev server and `vite preview` set these headers themselves (see `vite.config.ts`),
but production hosting has to set them independently — that config does not travel with the
build.

**Always test the production build, not just the dev server.** Several of the traps in this
stack pass in dev and fail only in the built artefact: workers compile to a different module
format, `import.meta.url` resolves differently, and top-level await is handled differently.
Run `vite build && vite preview` and confirm `crossOriginIsolated === true` in the console
before trusting a deployment.

## Self-hosted behind a reverse proxy

Add the headers at the proxy, scoped to the app's path. For nginx:

```nginx
location /showcase/sdr {
    add_header Cross-Origin-Opener-Policy same-origin;
    add_header Cross-Origin-Embedder-Policy require-corp;
}
```

If a CDN sits in front of the origin, check that it is not stripping or overriding these
headers — that is a common reason they are set correctly at the origin and absent at the
browser.

## As an embedded showcase on a Next.js site

The app is designed to mount as a component rather than an iframe. Two things are needed on
the host:

**Scoped headers.** Add an entry to the `headers()` block in `next.config.mjs` for the app's
route:

```js
{
  source: '/showcase/sdr/:path*',
  headers: [
    { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
    { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
  ],
}
```

**Self-hosted fonts.** A page under `require-corp` cannot load a font via a bare CSS
`@import` from Google Fonts — that request is made in no-CORS mode and is blocked. If the
host site imports fonts that way, move them to `next/font`, which self-hosts them at build
time. This is a prerequisite for the isolated route to render its text at all, and it is an
independent performance win.

See `packages/react` for the mount component.

## Browser support

- **WebUSB** is Chromium-only. Firefox and WebKit have both declined to implement it, so the
  receiver runs only in Chromium-based browsers on desktop and Android.
- **WebGPU** is furthest along in Chromium; where it is unavailable the app runs without the
  displays rather than refusing to start, since no browser ships a software fallback adapter.
- **Cross-origin isolation** and **AudioWorklet** are available everywhere and are not the
  constraint.
