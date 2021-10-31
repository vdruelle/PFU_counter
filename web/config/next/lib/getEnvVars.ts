import { getbool, getenv } from '../../../lib/getenv'
import { getDomain } from '../../../lib/getDomain'

export function getEnvVars() {
  const BABEL_ENV = getenv('BABEL_ENV')
  const NODE_ENV = getenv('NODE_ENV')
  const PRODUCTION = NODE_ENV === 'production'
  const DOMAIN = getDomain()
  const WATCH_POLL = getbool('WATCH_POLL', false)
  const ENABLE_ESLINT = getbool('WEB_ENABLE_ESLINT')
  const ENABLE_TYPE_CHECKING = getbool('WEB_ENABLE_TYPE_CHECKING')

  const common = {
    BABEL_ENV,
    DOMAIN,
    ENABLE_ESLINT,
    ENABLE_TYPE_CHECKING,
    NODE_ENV,
    PRODUCTION,
    WATCH_POLL,
  }

  if (PRODUCTION) {
    return {
      ...common,
      ENABLE_SOURCE_MAPS: getbool('WEB_PROD_ENABLE_SOURCE_MAPS'),
    }
  }

  return {
    ...common,
    ENABLE_SOURCE_MAPS: true,
  }
}
