declare module 'next-optimized-images' {
  export type ImageFormat = 'jpeg' | 'png' | 'svg' | 'webp' | 'gif' | 'ico'

  export interface GetNextOptimizedImagesOptions {
    handleImages?: ImageFormat[] // ['jpeg', 'png', 'svg', 'webp', 'gif', 'ico']
    inlineImageLimit?: number // 8192
    imagesFolder?: string // `/images`
    imagesPublicPath?: string // `/_next/static/${imagesFolder}/`
    imagesOutputPath?: string // `static/${imagesFolder}/`
    imagesName?: string // '[name]-[hash].[ext]'
    removeOriginalExtension?: boolean // false
    optimizeImagesInDev?: boolean // false
    mozjpeg?: any
    optipng?: any
    pngquant?: any
    gifsicle?: any
    svgo?: any
    svgSpriteLoader?: any
    webp?: any
    imageTrace?: any
    responsive?: any
    defaultImageLoader?: string // 'img-loader'
    optimizeImages?: boolean // true
  }

  export default function getNextOptimizedImages(options: GetNextOptimizedImagesOptions)
}
