import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const workspace = process.argv[2];
if (!workspace) throw new Error('Usage: node restore-runtime-assets.mjs /path/to/workspace');
const root = path.resolve(workspace, 'animflow/public');
const data = await readFile(path.join(root, 'motions/kimodo/index.json'));
if (data.length > 512 * 1024) throw new Error('Index exceeds size limit');
const index = JSON.parse(data);
if (index.version !== 1 || !Array.isArray(index.assets) || index.assets.length > 200) throw new Error('Invalid motion index');
const sha = (payload) => createHash('sha256').update(payload).digest('hex');
let restored = 0, cached = 0;
for (const asset of index.assets) {
  if (!/^\/motions\/kimodo\/[a-z0-9-]{1,128}\.bvh$/.test(asset.assetUrl) || !/^[a-f0-9]{64}$/.test(asset.sha256)) throw new Error('Invalid motion path/hash');
  const target = path.join(root, asset.assetUrl.slice(1));
  try {
    const existing = await readFile(target);
    if (sha(existing) !== asset.sha256) throw new Error(`Existing motion differs; not overwriting: ${target}`);
    cached++; continue;
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const response = await fetch(`https://animflow.app.nz${asset.assetUrl}`, { signal: AbortSignal.timeout(30000) });
  if (!response.ok) throw new Error(`Motion unavailable (${response.status}): ${asset.id}`);
  const reader = response.body.getReader();
  const chunks = []; let length = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    length += value.length;
    if (length > 4 * 1024 * 1024) { await reader.cancel(); throw new Error('Motion exceeds size limit'); }
    chunks.push(value);
  }
  const payload = Buffer.concat(chunks);
  if (sha(payload) !== asset.sha256) throw new Error(`Motion checksum mismatch: ${asset.id}`);
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, payload, { flag: 'wx' });
  restored++;
}
console.log(`Restored ${restored} checksum-verified motions; ${cached} already cached; no GPU or generation calls`);
