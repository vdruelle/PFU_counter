import { useFilePicker } from 'use-file-picker'

export function usePickFile(accept?: string | string[]) {
  const [openFileSelector, { plainFiles, clear }] = useFilePicker({
    accept,
    multiple: false,
    readFilesContent: false,
  })

  let file: File | undefined
  if (plainFiles.length > 0) {
    file = plainFiles[0] // eslint-disable-line prefer-destructuring
  }

  return { openFileSelector, file, clearFile: clear }
}
