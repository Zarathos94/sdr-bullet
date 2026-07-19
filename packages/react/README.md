# @sdr-bullet/react

A React component that mounts the browser SDR receiver, for embedding it as a showcase app
on a host site.

```tsx
import SdrReceiver from '@sdr-bullet/react'

export default function Page() {
  return <SdrReceiver />
}
```

## Host requirements

The receiver runs on `SharedArrayBuffer`, which is only available to a cross-origin isolated
document. The host has to supply that; the component cannot.

On a Next.js site, add a scoped `headers()` entry so the isolation applies only to this
route rather than the whole site — site-wide `require-corp` blocks any cross-origin
subresource that does not opt in, including fonts loaded via a CSS `@import`:

```js
// next.config.mjs
async headers() {
  return [
    {
      source: '/showcase/sdr/:path*',
      headers: [
        { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
        { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
      ],
    },
  ]
}
```

If the host site loads fonts through a bare `@import url(fonts.googleapis.com…)`, move them
to `next/font` first — that request is made in no-CORS mode and is blocked under
`require-corp`, so the isolated route would otherwise lose its fonts.

The receiver is Chromium-only, because WebUSB is. It degrades to no displays where WebGPU is
unavailable rather than refusing to start.

See [docs/deployment.md](../../docs/deployment.md) for the full picture.
