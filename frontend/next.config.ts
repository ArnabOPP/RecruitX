import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // An unrelated package-lock.json in the user's home directory otherwise
  // confuses Turbopack's workspace-root inference.
  turbopack: {
    root: path.join(__dirname),
  },
};

export default nextConfig;
