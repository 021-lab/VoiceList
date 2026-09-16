import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    include: ['tests/unit/**/*.test.js', 'tests/v2/**/*.test.js', 'tests/v2-client/**/*.test.js']
  }
});
