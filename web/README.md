# PFU counter web app

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


2. Prepare env variables

    ```
    cp .env.example .env
    ```
    
    Adjust `.env` if needed.


3. Install required Node.js modules (listed in `package.json`)

    ```
    yarn install
    ```

4. Run development server with hot reloading:

    ```bash
    yarn dev
    ```

5. Wait until the initial build is done and open `http://localhost:3000` in the browser.


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

## Creating a fake video device for testing

- Install required dependencies:

  ```bash
  sudo apt-get install ffmpeg v4l2loopback-dkms
  ```

- Create a fake webcam video device.

  ```bash
  sudo modprobe v4l2loopback card_label="Fake Webcam" exclusive_caps=1 devices=1 video_nr=10
  ```

  The parameter `card_label` is the name of the device. It is arbitrary. The parameter `video_nr=10` is the index of the created video device. The index `10` is selected arbitrarily. Adjust it you already have a video device with this index on your computer (`/dev/video10`), to avoid conficts. The video device will be destroyed on reboot. You will need to create it again.

- See what video devices there are:

  ```bash
  ls -1 /dev/video*
  ```

  Our new device should be under `/dev/video10`.

- Stream a video file to the fake webcam:

  ```bash
  ffmpeg -stream_loop -1 -re -i /path/to/video.mp4 -vcodec rawvideo -threads 0 -f v4l2 /dev/video10
  ```

- Stream an image file video file to the fake webcam:

  ```bash
  ffmpeg -loop 1 -re -i /path/tom/image.png -f v4l2 -vcodec rawvideo /dev/video10
  ```

  Keep this command running for webcam to keep streaming. The image or video will serve as a source of the content for the fake webcam. It will play this content in a loop. If you changed the parameter `video_nr` above, then you will need to use that value instead of `10` in `/dev/video10`.

- (Optional) Use `ffplay` to see what the fake webcam device is playing:

  ```bash
  ffplay /dev/video10
  ```

  It will show a window with the content that the webcam is streaming. If it works, then close this window. It is not needed for the webcam to function.

- Now the fake webcam is ready to be used in the web app. Refresh browser tab and the web app should now be able detect the fake webcam and disply the streamed content.
