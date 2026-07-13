#!/usr/bin/env node

import { resolveBuildProfileAppPath } from './release-evidence.mjs';

const profilePath = process.argv[2];
if (!profilePath || process.argv.length !== 3) {
  console.error('Usage: node scripts/resolve-build-profile-app.mjs <build-profile.json>');
  process.exit(2);
}

try {
  process.stdout.write(resolveBuildProfileAppPath(profilePath).appPath);
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
