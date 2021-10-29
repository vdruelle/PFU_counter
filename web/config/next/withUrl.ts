import type { NextConfig } from 'next'

import { addWebpackLoader } from './lib/addWebpackLoader'

export default function withUrl(nextConfig: NextConfig) {
  return addWebpackLoader(nextConfig, (webpackConfig, { dev, isServer }) => ({
    test: /\.(onnx)$/,
    exclude: nextConfig.exclude,
    use: [
      {
        loader: 'url-loader',
        options: {
          limit: false,
          name: dev ? '[name].[ext]' : '[name].[hash:7].[ext]',
          publicPath: `${nextConfig.assetPrefix}/_next/static/assets/`,
          outputPath: `${isServer ? '../' : ''}static/assets/`,
        },
      },
    ],
  }))
}
