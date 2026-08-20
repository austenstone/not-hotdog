import { cp, mkdir, readdir, rm } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = dirname(fileURLToPath(import.meta.url))
const webRoot = join(root, '..')

const copyDirectory = async ({ from, to, required }) => {
  let entries = []

  try {
    entries = await readdir(from)
  } catch (error) {
    if (!required && error?.code === 'ENOENT') {
      await mkdir(to, { recursive: true })
      return
    }

    throw error
  }

  await rm(to, { recursive: true, force: true })
  await mkdir(to, { recursive: true })

  await Promise.all(
    entries.map((entry) => cp(join(from, entry), join(to, entry), { recursive: true, force: true })),
  )
}

await copyDirectory({
  from: join(webRoot, 'node_modules', '@litertjs', 'core', 'wasm'),
  to: join(webRoot, 'public', 'wasm'),
  required: true,
})

await copyDirectory({
  from: join(webRoot, '..', 'models'),
  to: join(webRoot, 'public', 'models'),
  required: false,
})
