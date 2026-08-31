import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "../../../frontend/node_modules/playwright/index.mjs";

const outputDir = dirname(fileURLToPath(import.meta.url));
const baseUrl = "http://127.0.0.1:4173";

const scenarios = [
  {
    id: "01-capital-one-cet1",
    category: "Exact value from a canonical filing table",
    question:
      "What was Capital One Financial Corporation's CET1 capital ratio under the Basel III standardized approach on December 31, 2025?",
    expected: "14.3%",
    expectedPattern: /14\.3\s*(?:%|percent)/i,
  },
  {
    id: "02-capital-one-cyber-risk",
    category: "Narrative synthesis across filing evidence",
    question:
      "How does Capital One manage cybersecurity and technology risk under its enterprise risk framework?",
    expected:
      "Cybersecurity and technology risk are treated as operational risk within the enterprise framework and three-lines-of-defense model.",
    expectedPattern: /operational risk/i,
  },
  {
    id: "03-citi-jpm-operational-risk",
    category: "Bank-owned multi-bank comparison",
    question:
      "How do Citi and JPMorgan Chase each define operational risk in their 2025 Form 10-K filings?",
    expected:
      "Separate, cited definitions for Citi and JPMorgan Chase, including failed processes or systems, human factors/errors, and external events.",
    expectedPattern: /Citi[\s\S]*JPMorgan|JPMorgan[\s\S]*Citi/i,
  },
  {
    id: "04-bac-citi-cet1-comparison",
    category: "Exact-value multi-bank comparison",
    question:
      "What were the Standardized CET1 capital ratios for Bank of America Corporation and Citigroup on December 31, 2025?",
    expected: "Bank of America Corporation: 11.4%; Citigroup: 13.18%.",
    expectedPattern: /11\.4\s*(?:%|percent)[\s\S]*13\.18\s*(?:%|percent)|13\.18\s*(?:%|percent)[\s\S]*11\.4\s*(?:%|percent)/i,
  },
];

await mkdir(outputDir, { recursive: true });

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: { width: 1600, height: 1100 },
  deviceScaleFactor: 1,
  colorScheme: "light",
});
const page = await context.newPage();
page.setDefaultTimeout(240_000);

const results = [];
const reuseExisting = process.argv.includes("--reuse-existing");
const onlyId = process.argv.find((argument) => argument.startsWith("--only="))?.slice(7);
const explicitThreadUrl = process.argv.find((argument) => argument.startsWith("--thread-url="))?.slice(13);
const previousResults = reuseExisting || onlyId
  ? JSON.parse(await readFile(join(outputDir, "capture-results.json"), "utf8"))
  : [];
const scenariosToRun = onlyId ? scenarios.filter((scenario) => scenario.id === onlyId) : scenarios;
if (onlyId && scenariosToRun.length === 0) throw new Error(`Unknown scenario: ${onlyId}`);

async function prepareCaptureLayout() {
  await page.addStyleTag({
    content: `
      *, *::before, *::after { animation: none !important; transition: none !important; }
      html, body, #root, .app-shell { height: auto !important; min-height: 100% !important; overflow: visible !important; }
      .workspace { display: block !important; height: auto !important; min-height: 0 !important; padding: 20px !important; }
      .thread-sidebar, .conversation-composer, .scroll-bottom { display: none !important; }
      .workspace-main, .conversation, .conversation-scroll { height: auto !important; min-height: 0 !important; overflow: visible !important; }
      .conversation-list { width: min(100% - 48px, 980px) !important; padding: 36px 0 !important; }
    `,
  });
}

try {
  for (const [index, scenario] of scenariosToRun.entries()) {
    const previous = previousResults.find((result) => result.id === scenario.id);
    if (reuseExisting) {
      const threadUrl = previous?.thread_url ?? explicitThreadUrl;
      if (!threadUrl) throw new Error(`Missing saved thread URL for ${scenario.id}`);
      await page.goto(threadUrl, { waitUntil: "networkidle" });
    } else {
      await page.goto(baseUrl, { waitUntil: "networkidle" });
      await page.getByRole("button", { name: "New conversation" }).click();

      const composer = page.getByRole("textbox", { name: "Research question" });
      await composer.waitFor({ state: "visible" });
      await composer.fill(scenario.question);
      await page.getByRole("button", { name: "Send question" }).click();
    }

    const answerBody = page.locator(".turn").last().locator(".answer-body");
    await answerBody.waitFor({ state: "visible", timeout: 240_000 });
    await page.waitForTimeout(1000);

    const turn = page.locator(".turn").last();
    const answerText = (await turn.locator(".answer-text").first().innerText()).trim();
    const turnText = (await turn.innerText()).trim();
    const passedExpectedCheck = scenario.expectedPattern.test(turnText);

    await prepareCaptureLayout();

    await turn.screenshot({
      path: join(outputDir, `${scenario.id}-answer.png`),
      animations: "disabled",
    });

    const auditButton = turn.getByRole("button", { name: /^Evidence audit:/ });
    if (await auditButton.count()) {
      await auditButton.click();
      await turn.getByRole("region", { name: "Evidence audit details" }).waitFor();
    }

    const diagnosticsButton = turn.getByRole("button", { name: "Diagnostics" });
    await diagnosticsButton.click();
    const diagnosticsRegion = turn.getByRole("region", { name: "Diagnostics details" });
    await diagnosticsRegion.waitFor();
    const diagnosticsText = (await diagnosticsRegion.innerText()).trim();

    await turn.screenshot({
      path: join(outputDir, `${scenario.id}-diagnostics.png`),
      animations: "disabled",
    });

    let sourceText = null;
    const sourcesButton = turn.getByRole("button", { name: "Sources", exact: true });
    if (index === 0 && (await sourcesButton.count())) {
      await sourcesButton.click();
      const sourcePanel = page.locator(".source-panel");
      await sourcePanel.waitFor({ state: "visible" });
      await page.waitForTimeout(800);
      sourceText = (await sourcePanel.innerText()).trim();
      await page.screenshot({
        path: join(outputDir, `${scenario.id}-source-evidence.png`),
        fullPage: false,
        animations: "disabled",
      });
      await page.keyboard.press("Escape");
    }

    results.push({
      id: scenario.id,
      category: scenario.category,
      question: scenario.question,
      expected: scenario.expected,
      expected_check_passed: passedExpectedCheck,
      actual_answer: answerText,
      diagnostics: diagnosticsText,
      source_preview: sourceText,
      captured_at: new Date().toISOString(),
      thread_url: page.url(),
    });
  }
} finally {
  await browser.close();
}

const finalResults = onlyId
  ? scenarios
      .map((scenario) => results.find((result) => result.id === scenario.id)
        ?? previousResults.find((result) => result.id === scenario.id))
      .filter(Boolean)
  : results;
await writeFile(join(outputDir, "capture-results.json"), `${JSON.stringify(finalResults, null, 2)}\n`, "utf8");

if (results.some((result) => !result.expected_check_passed)) {
  process.exitCode = 2;
}
