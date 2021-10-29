import path from 'path'

import { uniq } from 'lodash'

import getWithMDX from '@next/mdx'
import withPlugins from 'next-compose-plugins'
import getWithTranspileModules from 'next-transpile-modules'
import getNextOptimizedImages from 'next-optimized-images'
import intercept from 'intercept-stdout'

import { findModuleRoot } from '../../lib/findModuleRoot'
import { getGitBranch } from '../../lib/getGitBranch'
import { getBuildNumber } from '../../lib/getBuildNumber'
import { getBuildUrl } from '../../lib/getBuildUrl'
import { getGitCommitHash } from '../../lib/getGitCommitHash'
import { getEnvVars } from './lib/getEnvVars'

import getWithFriendlyConsole from './withFriendlyConsole'
import getWithTypeChecking from './withTypeChecking'
import withFriendlyChunkNames from './withFriendlyChunkNames'
import withIgnore from './withIgnore'
import withResolve from './withResolve'
import withWebpackWatchPoll from './withWebpackWatchPoll'
import { getWithRobotsTxt } from './withRobotsTxt'
import { getWithCopy } from './withCopy'
import withUrl from './withUrl'

// Ignore recoil warning messages in stdout
// https://github.com/facebookexperimental/Recoil/issues/733
intercept((text: string) => (text.includes('Duplicate atom key') ? '' : text))

const { DOMAIN, ENABLE_ESLINT, ENABLE_SOURCE_MAPS, ENABLE_TYPE_CHECKING, PRODUCTION, WATCH_POLL } = getEnvVars()

const BRANCH_NAME = getGitBranch()

const { pkg, moduleRoot } = findModuleRoot()

const clientEnv = {
  BRANCH_NAME,
  PACKAGE_VERSION: pkg.version ?? '',
  BUILD_NUMBER: getBuildNumber(),
  BUILD_URL: getBuildUrl(),
  COMMIT_HASH: getGitCommitHash(),
  DOMAIN,
}

console.info(`Client-side Environment:\n${JSON.stringify(clientEnv, null, 2)}`)

const nextConfig = {
  distDir: `.build/${process.env.NODE_ENV}/tmp`,
  pageExtensions: ['js', 'jsx', 'ts', 'tsx', 'md', 'mdx'],
  poweredByHeader: false,
  onDemandEntries: {
    maxInactiveAge: 60 * 1000,
    pagesBufferLength: 2,
  },
  future: {
    excludeDefaultMomentLocales: true,
  },
  devIndicators: {
    buildActivity: false,
    autoPrerender: false,
  },
  typescript: {
    ignoreDevErrors: true,
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  env: clientEnv,
  experimental: {
    esmExternals: true,
  },
  images: {
    disableStaticImages: true,
  },
  productionBrowserSourceMaps: ENABLE_SOURCE_MAPS,
}

const withMDX = getWithMDX({
  extension: /\.mdx?$/,
  options: {
    remarkPlugins: [
      // prettier-ignore
      require('remark-breaks'),
      require('remark-images'),
      require('remark-math'),
      require('remark-slug'),
      [
        require('remark-toc'),
        {
          tight: true,
        },
      ],
    ],
    rehypePlugins: [],
  },
})

const withFriendlyConsole = getWithFriendlyConsole({
  clearConsole: false,
  projectRoot: path.resolve(moduleRoot),
  packageName: pkg.name || 'web',
  progressBarColor: '#51628F',
})

// const withStaticComprression = getWithStaticCompression({ brotli: false })

const withTypeChecking = getWithTypeChecking({
  typeChecking: ENABLE_TYPE_CHECKING,
  eslint: ENABLE_ESLINT,
  memoryLimit: 2048,
})

const transpilationListDev: string[] = [
  // prettier-ignore
  'debug',
  'lodash',
]

const transpilationListProd = uniq([
  ...transpilationListDev,
  // prettier-ignore
])

const withTranspileModules = getWithTranspileModules(PRODUCTION ? transpilationListProd : transpilationListDev)

const withRobotsTxt = getWithRobotsTxt(`User-agent: *\nDisallow:${BRANCH_NAME === 'release' ? '' : ' *'}\n`)

const withImages = getNextOptimizedImages({
  handleImages: ['png'],
  removeOriginalExtension: true,
  optimizeImagesInDev: true,
  inlineImageLimit: 32768,
  mozjpeg: {
    quality: 80,
  },
  optipng: {
    interlaced: true,
    optimizationLevel: 3,
  },
  pngquant: false,
  gifsicle: {
    interlaced: true,
    optimizationLevel: 3,
  },
  svgo: false,
  svgSpriteLoader: false,
  webp: {
    preset: 'default',
    quality: 75,
  },
  responsive: false,
})

const withCopy = getWithCopy({
  patterns: [
    // Copy 'onnxruntime-web' wasm files to the output dir
    {
      context: path.join(moduleRoot, 'node_modules/onnxruntime-web/dist'),
      from: `*.wasm`,
      to: 'static/chunks/[name][ext]',
      toType: 'template',
    },
  ],
})

const config = withPlugins(
  [
    [withImages],
    [withIgnore],
    [withUrl],
    [withFriendlyConsole],
    [withMDX],
    [withTypeChecking],
    [withTranspileModules],
    WATCH_POLL && [withWebpackWatchPoll],
    [withFriendlyChunkNames],
    [withResolve],
    [withRobotsTxt],
    [withCopy],
  ].filter(Boolean),
  nextConfig,
)

export default config
