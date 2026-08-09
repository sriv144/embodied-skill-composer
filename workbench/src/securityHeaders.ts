const documentContentSecurityPolicy = [
  "default-src 'self'",
  "script-src 'self' 'wasm-unsafe-eval'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "connect-src 'self'",
  "worker-src 'self' blob:",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'self'",
  "frame-ancestors 'none'"
].join("; ");

const commonSecurityHeaders = {
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
};

export const developmentSecurityHeaders = {
  ...commonSecurityHeaders,
  // Vite injects its React-refresh preamble as an inline module in dev only.
  "Content-Security-Policy": documentContentSecurityPolicy.replace(
    "script-src 'self' 'wasm-unsafe-eval'",
    "script-src 'self' 'wasm-unsafe-eval' 'unsafe-inline'"
  )
};

export const previewSecurityHeaders = {
  ...commonSecurityHeaders,
  "Content-Security-Policy": documentContentSecurityPolicy
};
