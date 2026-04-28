/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    const target =
      process.env.EASYTRAIN_SERVER_URL ?? "http://127.0.0.1:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${target}/:path*`,
      },
    ];
  },
};

export default nextConfig;
