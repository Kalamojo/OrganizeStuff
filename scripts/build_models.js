const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('--- Preparing backend models ---');

// Ensure output dir
const modelDir = path.join(__dirname, '..', 'api', 'clip_model');
fs.mkdirSync(modelDir, { recursive: true });

// Install Python build deps (Vercel Node has Python access)
const python = spawnSync('python', ['-m', 'pip', 'install', '-r', 'backend/requirements-build.txt'], {
  stdio: 'inherit',
  timeout: 300000  // 5min
});

// Run your model prep script
console.log('Running prepare_model.py...');
const result = spawnSync('python', ['scripts/prepare_model.py', 'api'], {
  stdio: 'inherit',
  timeout: 600000  // 10min for model download
});

if (result.error) {
  console.error('Model build failed:', result.error);
  process.exit(1);
}

console.log('✅ Models ready at api/clip_model/');
console.log('Files:', fs.readdirSync(modelDir));
