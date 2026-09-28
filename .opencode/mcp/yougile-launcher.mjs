#!/usr/bin/env node
// Launcher for the YouGile MCP server (@nebelov/yougile-mcp), referenced by opencode.jsonc.
// MCP clients do not read .env files, so this wrapper loads the repo-root .env
// and passes the token to the server as YOUGILE_API_KEY.
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const ENV_FILE = resolve(REPO_ROOT, ".env");

function readDotEnv(file) {
  let text;
  try {
    const buf = readFileSync(file);
    text =
      buf[0] === 0xff && buf[1] === 0xfe
        ? buf.toString("utf16le", 2)
        : buf.toString("utf8").replace(/^\uFEFF/, "");
  } catch {
    return {};
  }
  const vars = {};
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq <= 0) continue;
    const key = line.slice(0, eq).trim();
    let value = line.slice(eq + 1).trim();
    const quote = value[0];
    if (
      value.length >= 2 &&
      (quote === '"' || quote === "'") &&
      value.endsWith(quote)
    ) {
      value = value.slice(1, -1);
    }
    vars[key] = value;
  }
  return vars;
}

const dotenv = readDotEnv(ENV_FILE);
const apiKey =
  dotenv.YOUGILE_API_TOKEN ||
  process.env.YOUGILE_API_TOKEN ||
  process.env.YOUGILE_API_KEY;

if (!apiKey) {
  console.error(
    `[yougile-mcp] YOUGILE_API_TOKEN is not set. Add it to .env (${ENV_FILE}).`,
  );
}

const env = { ...process.env };
if (apiKey) env.YOUGILE_API_KEY = apiKey;
if (dotenv.YOUGILE_COMPANY_ID) env.YOUGILE_COMPANY_ID = dotenv.YOUGILE_COMPANY_ID;

const child = spawn("npx", ["-y", "@nebelov/yougile-mcp"], {
  env,
  stdio: "inherit",
  shell: true,
});
child.on("exit", (code, signal) => process.exit(signal ? 1 : (code ?? 1)));
