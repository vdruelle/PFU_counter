import 'reflect-metadata'
import 'resize-observer-polyfill/dist/ResizeObserver.global'

import 'css.escape'

import dynamic from 'next/dynamic'
import React, { useMemo } from 'react'
import { RecoilRoot } from 'recoil'
import { QueryClient, QueryClientProvider } from 'react-query'
import { ReactQueryDevtools } from 'react-query/devtools'

import type { AppProps } from 'next/app'
import { ThemeProvider } from 'styled-components'
import { MDXProvider } from '@mdx-js/react'

import { theme } from 'src/theme'

import 'src/styles/global.scss'
import { memoize } from 'lodash'

const mdxComponents = {}

// Ignore recoil warning messages in browser console
// https://github.com/facebookexperimental/Recoil/issues/733
const mutedConsole = memoize((console: Console) => ({
  ...console,
  warn: (...args: string[]) => (args[0].includes('Duplicate atom key') ? null : console.warn(...args)),
}))
global.console = mutedConsole(global.console)

function MyApp({ Component, pageProps, router }: AppProps) {
  const queryClient = useMemo(() => new QueryClient(), [])

  return (
    <RecoilRoot>
      <ThemeProvider theme={theme}>
        <MDXProvider components={(components) => ({ ...components, ...mdxComponents })}>
          <QueryClientProvider client={queryClient}>
            <Component {...pageProps} />
            <ReactQueryDevtools initialIsOpen={false} />
          </QueryClientProvider>
        </MDXProvider>
      </ThemeProvider>
    </RecoilRoot>
  )
}

export default dynamic(() => Promise.resolve(MyApp), { ssr: false })
