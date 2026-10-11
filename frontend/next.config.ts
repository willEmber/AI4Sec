import type { NextConfig } from "next";

const backendUrl = process.env.BACKEND_URL || "http://localhost:8001";

// Where images may be loaded from (P13 §5.1). The Markdown renderer already
// refuses images that are not ours; this is the second lock, for the case the
// first one has a hole: a model that read planted text must not be able to make
// the reader's browser call an address of the text's choosing. Only `img-src`
// is set, so nothing else about the page is restricted.
// Login providers serve avatars from their own hosts; `CSP_IMG_SRC` adds more.
const imgSrc = [
  "'self'",
  "data:",
  "blob:",
  "https://avatars.githubusercontent.com",
  "https://*.googleusercontent.com",
  "https://linux.do",
  "https://*.linux.do",
  ...(process.env.CSP_IMG_SRC || "").split(/[\s,]+/).filter(Boolean),
].join(" ");

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    middlewareClientMaxBodySize: 100 * 1024 * 1024, // 100MB for PDF uploads
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [{ key: "Content-Security-Policy", value: `img-src ${imgSrc}` }],
      },
    ];
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
  webpack: (config) => {
    // Handle canvas for react-pdf
    config.resolve.alias.canvas = false;
    return config;
  },
};

export default nextConfig;
