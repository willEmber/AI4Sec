import type { NextConfig } from "next";

const backendUrl = process.env.BACKEND_URL || "http://localhost:8001";

const nextConfig: NextConfig = {
  output: "standalone",
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
