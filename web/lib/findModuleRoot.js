/* eslint-disable no-param-reassign */
const fs = require('fs-extra')
const path = require('path')

module.exports = {
  findModuleRoot(maxDepth = 10) {
    let moduleRoot = __dirname
    while (--maxDepth) {
      moduleRoot = path.resolve(moduleRoot, '..')
      const file = path.join(moduleRoot, 'package.json')
      if (fs.existsSync(file)) {
        const pkg = fs.readJsonSync(file)
        return { moduleRoot, pkg }
      }
    }
    throw new Error('Module root not found')
  },
}
