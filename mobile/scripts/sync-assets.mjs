import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const mobileRoot = join(dirname(fileURLToPath(import.meta.url)), '..')
const repoRoot = join(mobileRoot, '..')
const sourceModelsDir = join(repoRoot, 'models')
const bundledModelsDir = join(mobileRoot, 'src', 'assets', 'models')
const generatedDir = join(mobileRoot, 'src', 'generated')
const generatedFile = join(generatedDir, 'modelAssets.ts')

const readJsonIfPresent = async (path) => {
  try {
    return JSON.parse(await readFile(path, 'utf8'))
  } catch (error) {
    if (error?.code === 'ENOENT') {
      return null
    }

    throw error
  }
}

const copyModelsIfPresent = async () => {
  let entries = []

  try {
    entries = await readdir(sourceModelsDir)
  } catch (error) {
    if (error?.code === 'ENOENT') {
      await mkdir(bundledModelsDir, { recursive: true })
      return []
    }

    throw error
  }

  await rm(bundledModelsDir, { recursive: true, force: true })
  await mkdir(bundledModelsDir, { recursive: true })
  await Promise.all(entries.map((entry) => cp(join(sourceModelsDir, entry), join(bundledModelsDir, entry), { recursive: true, force: true })))

  return entries
}

const writeGeneratedAssets = async () => {
  const entries = await copyModelsIfPresent()
  const metadata = await readJsonIfPresent(join(sourceModelsDir, 'metadata.json'))
  const modelName = typeof metadata?.model === 'string' ? metadata.model : null
  const hasModel = modelName ? entries.includes(modelName) : false
  const modelRequire = hasModel ? `require(${JSON.stringify(`../assets/models/${modelName}`)}) as number` : 'null'
  const metadataLiteral = metadata ? JSON.stringify(metadata, null, 2) : 'null'

  await mkdir(generatedDir, { recursive: true })
  await writeFile(
    generatedFile,
    `export const bundledModelAsset: number | null = ${modelRequire}\n\nexport const bundledMetadata = ${metadataLiteral} as unknown\n`,
  )
}

await writeGeneratedAssets()
