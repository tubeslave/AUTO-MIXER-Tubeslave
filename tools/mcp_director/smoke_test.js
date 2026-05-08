#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawn, spawnSync } = require("node:child_process");

const repo = fs.mkdtempSync(path.join(os.tmpdir(), "automixer-mcp-director-"));
spawnSync("git", ["init", "-b", "feature/mcp-director"], { cwd: repo, encoding: "utf8" });

const serverPath = path.join(__dirname, "server.js");
const server = spawn(process.execPath, [serverPath], {
  env: {
    ...process.env,
    MCP_DIRECTOR_REPO: repo,
    MCP_DIRECTOR_REQUIRED_BRANCH: "feature/mcp-director"
  },
  stdio: ["pipe", "pipe", "pipe"]
});

let stdout = Buffer.alloc(0);
const responses = [];

server.stdout.on("data", (chunk) => {
  stdout = Buffer.concat([stdout, chunk]);
  while (true) {
    const parsed = readFrame(stdout);
    if (!parsed) {
      break;
    }
    stdout = stdout.subarray(parsed.nextOffset);
    responses.push(parsed.message);
  }
});

server.stderr.on("data", (chunk) => {
  process.stderr.write(chunk);
});

function send(id, method, params = {}) {
  const message = { jsonrpc: "2.0", id, method, params };
  const body = JSON.stringify(message);
  server.stdin.write(`Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`);
}

function readFrame(buffer) {
  const index = buffer.indexOf("\r\n\r\n");
  if (index < 0) {
    return null;
  }
  const header = buffer.subarray(0, index).toString("utf8");
  const match = /^Content-Length:\s*(\d+)\s*$/im.exec(header);
  if (!match) {
    throw new Error(`Missing Content-Length in response: ${header}`);
  }
  const length = Number(match[1]);
  const start = index + 4;
  const end = start + length;
  if (buffer.length < end) {
    return null;
  }
  return {
    message: JSON.parse(buffer.subarray(start, end).toString("utf8")),
    nextOffset: end
  };
}

function waitForResponses(count, timeoutMs = 5000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const timer = setInterval(() => {
      if (responses.length >= count) {
        clearInterval(timer);
        resolve();
      } else if (Date.now() - started > timeoutMs) {
        clearInterval(timer);
        reject(new Error(`Timed out waiting for ${count} responses; got ${responses.length}`));
      }
    }, 25);
  });
}

(async () => {
  send(1, "initialize", { protocolVersion: "2024-11-05", clientInfo: { name: "smoke", version: "0" } });
  send(2, "tools/list");
  send(3, "tools/call", {
    name: "create_task",
    arguments: {
      title: "Smoke test task",
      description: "Validate task persistence.",
      agent: "analyzer"
    }
  });
  send(4, "tools/call", {
    name: "git_report",
    arguments: {
      title: "Smoke Git Report"
    }
  });
  send(5, "tools/call", {
    name: "telegram_notify",
    arguments: {
      message: "Smoke notification",
      dryRun: true
    }
  });

  await waitForResponses(5);
  server.stdin.end();
  server.kill();

  const errors = responses.filter((response) => response.error);
  if (errors.length) {
    throw new Error(`MCP errors: ${JSON.stringify(errors, null, 2)}`);
  }

  const tools = responses[1].result.tools.map((tool) => tool.name).sort();
  const expected = ["create_task", "git_report", "run_agent", "run_full_review", "telegram_notify"].sort();
  if (JSON.stringify(tools) !== JSON.stringify(expected)) {
    throw new Error(`Unexpected tools: ${tools.join(", ")}`);
  }

  const taskFiles = fs.readdirSync(path.join(repo, ".paperclip", "tasks")).filter((file) => file.endsWith(".json"));
  const reportFiles = fs.readdirSync(path.join(repo, ".paperclip", "reports")).filter((file) => file.endsWith(".md"));
  if (taskFiles.length !== 1) {
    throw new Error(`Expected one task file, got ${taskFiles.length}`);
  }
  if (reportFiles.length < 2) {
    throw new Error(`Expected at least two report files, got ${reportFiles.length}`);
  }

  process.stdout.write(`MCP Director smoke test passed in ${repo}\n`);
})().catch((error) => {
  server.kill();
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exit(1);
});
