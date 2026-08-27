import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.route("**/api/threads", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ threads: [] }) });
  });
  await page.goto("/");
});

test("welcome composer and navigation have no automatically detectable serious violations", async ({ page }) => {
  await expect(page.getByRole("textbox", { name: "Research question" })).toBeVisible();
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations.filter((violation) => violation.impact === "serious" || violation.impact === "critical")).toEqual([]);
});

test("document picker is reachable and operable from the keyboard", async ({ page }) => {
  await page.getByRole("button", { name: "Upload file" }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("dialog", { name: "Upload Document" })).toBeVisible();
  const picker = page.getByRole("button", { name: /Drag & drop file here/i });
  await expect(picker).toBeFocused();
  await expect(picker).toHaveAttribute("aria-describedby");
});
