# web app

## Development

1. Install dependencies (once)

 - install, nvm: https://github.com/nvm-sh/nvm

 - install Node.js version mentioned in `.nvmrc` file:

    ```bash
    cd web/
    nvm use
    ```

 - verify the installation: `node --version` should match the version in `.nvmrc` file.

 - install yarn and nodemon globally

   ```bash
   npm install -g yarn@1.22.17 nodemon@2.0.13
   ```

 - verify the installation: `yarn` and `nodemon` should be in `$PATH` and `yarn --version` and `nodemon --version` should match the versions above.


3. Prepare env variables

```
cp .env.example .env
```

Adjust `.env` if needed.


4. Install required Node.js modules (listed in `package.json`)

```
yarn install
```

5. Run development server with hot reloading:

```bash
yarn dev
```

6. Wait until the initial build is done and open `http://localhost:3000` in the browser.


## Production build

Static HTML/CSS/JS can be generated using

```bash
yarn prod:build
```

The resulting files will be in `web/.build/production/web`. They can be deployed to any static server.

Locally they can be tested with

```bash
yarn prod:serve
```

This will run a server on `http://localhost:8080`


## Code formatting

```bash
yarn format:fix
```

## Linting (static analysis)

```bash
yarn lint
```

Some of the issues can be automatically fixed with

```bash
yarn lint:fix
```
