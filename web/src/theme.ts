import { rgba } from 'polished'

const gridBreakpoints = {
  xs: '0',
  sm: '576px',
  md: '768px',
  lg: '992px',
  xl: '1200px',
  xxl: '2000px',
}

const containerMaxWidths = {
  sm: '540px',
  md: '720px',
  lg: '960px',
  xl: '1140px',
  xxl: '1950px',
}

export const white = '#fff'
export const gray100 = '#f8f9fa'
export const gray200 = '#e9ecef'
export const gray300 = '#dee2e6'
export const gray400 = '#ced4da'
export const gray500 = '#adb5bd'
export const gray600 = '#868e96'
export const gray700 = '#495057'
export const gray800 = '#373a3c'
export const gray900 = '#212529'
export const black = '#000'

export const blue = '#2780e3'
export const indigo = '#6610f2'
export const purple = '#613d7c'
export const pink = '#e83e8c'
export const red = '#ff0039'
export const orange = '#f0ad4e'
export const yellow = '#ff7518'
export const green = '#3fb618'
export const teal = '#20c997'
export const cyan = '#9954bb'

export const primary = blue
export const secondary = gray800
export const success = green
export const info = cyan
export const warning = yellow
export const danger = red
export const light = gray100
export const dark = gray800

export const basicColors = {
  white,
  gray100,
  gray200,
  gray300,
  gray400,
  gray500,
  gray600,
  gray700,
  gray800,
  gray900,
  black,
  blue,
  indigo,
  purple,
  pink,
  red,
  orange,
  yellow,
  green,
  teal,
  cyan,
}

export const themeColors = {
  primary,
  secondary,
  success,
  info,
  warning,
  danger,
  light,
  dark,
}

export const font = {
  sansSerif: `'Roboto', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Helvetica Neue', 'Arial', 'system-ui', 'system-sans', 'sans-serif'`,
  monospace: `'Roboto Mono', 'Menlo', 'system-mono'`,
  default: 'sans-serif',
}

export const shadows = {
  lighter: `1px 1px 1px 1px ${rgba(gray500, 0.2)}`,
  light: `1px 1px 2px 2px ${rgba(gray600, 0.2)}`,
  slight: `2px 2px 2px 2px ${rgba(gray700, 0.25)}`,
  medium: `2px 2px 3px 3px ${rgba(gray700, 0.25)}`,
  normal: `2px 2px 3px 3px ${rgba(gray900, 0.25)}`,
  thick: `3px 3px 3px 5px ${rgba(gray900, 0.33)}`,
  filter: {
    slight: `1px 1px 1px ${rgba(gray700, 0.25)}`,
    medium: `2px 2px 3px ${rgba(gray900, 0.33)}`,
  },
}

const uploadZoneBoxShadow = `1px 1px 3px 3px ${rgba(gray700, 0.25)}`

export const uploadZone = {
  'background': {
    normal: basicColors.gray100,
    accept: rgba(basicColors.gray700, 0.25),
    reject: rgba(themeColors.danger, 0.25),
    hover: basicColors.white,
  },
  'color': {
    normal: basicColors.gray700,
    accept: themeColors.success,
    reject: themeColors.danger,
    hover: basicColors.gray900,
  },
  'border': {
    normal: `4px ${rgba(basicColors.gray500, 0)} dashed`,
    accept: `4px ${rgba(basicColors.gray700, 0.75)} dashed`,
    reject: `4px ${rgba(themeColors.danger, 0.75)} dashed`,
    hover: `4px ${rgba(basicColors.gray600, 0.75)} dashed`,
  },
  'box-shadow': {
    normal: 'none',
    accept: uploadZoneBoxShadow,
    reject: uploadZoneBoxShadow,
    hover: uploadZoneBoxShadow,
  },
}

export const theme = {
  ...basicColors,
  ...themeColors,
  ...gridBreakpoints,
  containerMaxWidths,
  font,
  shadows,
  uploadZone,
}

export type Theme = typeof theme
