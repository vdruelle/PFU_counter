/**
 * Serves production build artifacts.
 *
 * /!\ Only for development purposes, e.g. verifying that production build runs
 * on developer's machine.
 *
 * This server is very naive, slow and insecure. Real-world deployments should
 * use either a 3rd-party static hosting or a robust static server, such as
 * Nginx, instead.
 *
 */

import type { ServerResponse } from 'http'
import path from 'path'

import express from 'express'

import allowMethods from 'allow-methods'
import expressStaticGzip from 'express-static-gzip'

import { getenv } from '../lib/getenv'
import { findModuleRoot } from '../lib/findModuleRoot'

const { moduleRoot } = findModuleRoot()

const buildDir = path.join(moduleRoot, '.build', 'production', 'web')
const nextDir = path.join(buildDir, '_next')

export interface NewHeaders {
  [key: string]: { key: string; value: string }[]
}

function main() {
  const app = express()

  const expressStaticGzipOptions = { enableBrotli: true, serveStatic: { extensions: ['html'] } }

  const cacheNone = {
    ...expressStaticGzipOptions,
    serveStatic: {
      ...expressStaticGzipOptions.serveStatic,
      setHeaders: (res: ServerResponse) => res.setHeader('Cache-Control', 'no-cache'),
    },
  }
  const cacheOneYear = {
    ...expressStaticGzipOptions,
    serveStatic: {
      ...expressStaticGzipOptions.serveStatic,
      maxAge: '31556952000',
      immutable: true,
    },
  }

  app.use(allowMethods(['GET', 'HEAD']))
  app.use('/_next', expressStaticGzip(nextDir, cacheOneYear))
  app.get('*', expressStaticGzip(buildDir, cacheNone))

  const port = getenv('WEB_PORT_PROD')
  app.listen(port, () => {
    console.info(`Server is listening on port ${port}`)
  })
}

main()
