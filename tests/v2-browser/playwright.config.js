import { defineConfig } from '@playwright/test';
export default defineConfig({ testDir: '.', testMatch: '**/*.spec.js', workers: 1, timeout: 45000,
  use: { baseURL: process.env.V02_BASE_URL || 'http://127.0.0.1:4511/', trace: 'retain-on-failure', screenshot: 'only-on-failure' } });
