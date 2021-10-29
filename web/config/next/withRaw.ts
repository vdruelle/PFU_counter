import type { NextConfig } from 'next'

import { addWebpackLoader } from './lib/addWebpackLoader'

export default function withRaw(nextConfig: NextConfig) {
  return addWebpackLoader(nextConfig, (webpackConfig, { dev }) => ({
    test: /\.(txt|csv|tsv|bin)$/i,
    use: [
      {
        loader: 'raw-loader',
      },
    ],
  }))
}
