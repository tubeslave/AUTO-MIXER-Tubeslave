#!/usr/bin/env node
"use strict";

import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const server = spawn(process.execPath, [path.join(here, "server.js")], {
  env: { ...process.env, AUTOMIXER_WING_LAB_AUTOWRITE: "0" },
  stdio: ["pipe", "pipe", "inherit"]
});

let stdout = Buffer.alloc(0);
const responses = [];

server.stdout.on("data", (chunk) => {
  stdout = Buffer.concat([stdout, chunk]);
  while (true) {
    const headerEnd = stdout.indexOf("\r\n\r\n");
    if (headerEnd < 0) return;
    const header = stdout.subarray(0, headerEnd).toString("utf8");
    const match = /^Content-Length:\s*(\d+)\s*$/im.exec(header);
    if (!match) throw new Error("missing_content_length");
    const length = Number(match[1]);
    const start = headerEnd + 4;
    if (stdout.length < start + length) return;
    responses.push(JSON.parse(stdout.subarray(start, start + length).toString("utf8")));
    stdout = stdout.subarray(start + length);
  }
});

function send(id, method, params = {}) {
  const body = JSON.stringify({ jsonrpc: "2.0", id, method, params });
  server.stdin.write(`Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`);
}

function waitFor(count, timeoutMs = 5000) {
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const timer = setInterval(() => {
      if (responses.length >= count) {
        clearInterval(timer);
        resolve();
      } else if (Date.now() - started > timeoutMs) {
        clearInterval(timer);
        reject(new Error(`timeout_waiting_for_${count}_responses`));
      }
    }, 20);
  });
}

try {
  send(1, "initialize", {
    protocolVersion: "2024-11-05",
    clientInfo: { name: "smoke", version: "1" }
  });
  send(2, "tools/list");
  send(3, "tools/call", { name: "wing_lab_status", arguments: {} });
  send(4, "tools/call", { name: "automixer_pipeline_catalog", arguments: {} });
  await waitFor(4);

  const errors = responses.filter((response) => response.error);
  if (errors.length) throw new Error(JSON.stringify(errors));
  const names = responses[1].result.tools.map((tool) => tool.name).sort();
  const expected = [
    "automixer_pipeline_catalog",
    "wing_lab_apply_decisions",
    "wing_lab_arm",
    "wing_lab_emergency_stop",
    "wing_lab_status",
    "wing_lab_stop",
    "wing_lab_trace"
  ].sort();
  if (JSON.stringify(names) !== JSON.stringify(expected)) {
    throw new Error(`unexpected_tools:${names.join(",")}`);
  }
  const status = responses[2].result.structuredContent;
  if (status.enabled !== false || status.armed !== false) {
    throw new Error(`unsafe_default_status:${JSON.stringify(status)}`);
  }
  const catalog = responses[3].result.structuredContent;
  if (catalog.summary.experimental_live_execution_allowed !== false) {
    throw new Error("pipeline_catalog_did_not_fail_closed");
  }
  process.stdout.write("Automixer WING Lab MCP smoke test passed\n");
} finally {
  server.stdin.end();
  server.kill("SIGTERM");
}
