import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
    // Look for test files in the "tests" directory, relative to this configuration file.
    testDir: 'browser-tests',

    // Run all tests in parallel.
    fullyParallel: true,

    // Fail the build on CI if you accidentally left test.only in the source code.
    forbidOnly: !!process.env.CI,
    retries: 0,

    // Opt out of parallel tests on CI.
    workers: process.env.CI ? 1 : undefined,

    // Reporter to use
    reporter: 'html',

    use: {
        // Base URL to use in actions like `await page.goto('/')`.
        baseURL: 'http://localhost:5173/browser-tests/',

        // Collect trace when retrying the failed test.
        trace: 'on-first-retry',
    },
    // Configure projects for major browsers.
    projects: [
        {
            name: 'Google Chrome',
            use: {
                ...devices['Desktop Chrome'],
                channel: 'chrome',
                launchOptions: {
                    args: ['--ozone-platform=x11', '--enable-unsafe-webgpu', '--enable-features=Vulkan'],
                },
            },
        },
    ],
    // Run your local dev server before starting the tests.
    webServer: {
        command: 'npm run dev',
        url: 'http://localhost:5173/browser-tests/generate.html',
        reuseExistingServer: !process.env.CI,
    },
});
