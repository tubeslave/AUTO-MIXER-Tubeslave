#!/usr/bin/env node
"use strict";

import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import readline from "node:readline";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "../..");
const bridgePath = path.join(repo, "tools", "wing_lab_mcp_bridge.py");
const bridge = spawn("/usr/bin/python3", [bridgePath], {
  cwd: repo,
  env: {
    AUTOMIXER_WING_LAB_AUTOWRITE:
      process.env.AUTOMIXER_WING_LAB_AUTOWRITE || "0",
    HOME: process.env.HOME || "",
    LANG: process.env.LANG || "C.UTF-8"
  },
  stdio: ["pipe", "pipe", "inherit"]
});

let input = Buffer.alloc(0);
let nextBridgeId = 1;
const pending = new Map();
const bridgeLines = readline.createInterface({ input: bridge.stdout });

bridgeLines.on("line", (line) => {
  let response;
  try {
    response = JSON.parse(line);
  } catch (_error) {
    return;
  }
  const waiter = pending.get(response.id);
  if (waiter) {
    pending.delete(response.id);
    response.error
      ? waiter.reject(new Error(response.error.message))
      : waiter.resolve(response.result);
  }
});

bridge.on("exit", (code) => {
  const error = new Error(`wing_lab_bridge_exited:${code}`);
  for (const waiter of pending.values()) waiter.reject(error);
  pending.clear();
});

function bridgeCall(method, params = {}) {
  const id = nextBridgeId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    bridge.stdin.write(`${JSON.stringify({ id, method, params })}\n`);
  });
}

const tools = [
  {
    name: "automixer_pipeline_catalog",
    description: "List canonical, working, experimental, and duplicate pipeline entrypoints.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false }
  },
  {
    name: "wing_lab_status",
    description: "Show the local WING Rack laboratory autowrite state and correction scope.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false }
  },
  {
    name: "wing_lab_arm",
    description: "Arm one bounded WING Rack lab session. This is the single session-level write approval.",
    inputSchema: {
      type: "object",
      required: ["host", "console_model"],
      properties: {
        host: { type: "string", minLength: 1, maxLength: 255 },
        port: { type: "integer", minimum: 1, maximum: 65535, default: 2223 },
        console_model: { type: "string", const: "WING Rack" },
        duration_sec: { type: "integer", minimum: 30, maximum: 1800, default: 900 },
        max_corrections: { type: "integer", minimum: 1, maximum: 256, default: 128 }
      },
      additionalProperties: false
    }
  },
  {
    name: "wing_lab_apply_decisions",
    description: "Write accepted bounded Automixer decisions without per-decision confirmation, with preimage, readback, WAL, and rollback tracking.",
    inputSchema: {
      type: "object",
      required: ["decisions"],
      properties: {
        decisions: {
          type: "array",
          minItems: 1,
          maxItems: 256,
          items: {
            type: "object",
            required: [
              "id",
              "family",
              "accepted",
              "reason",
              "confidence",
              "value"
            ]
          }
        }
      },
      additionalProperties: false
    }
  },
  {
    name: "wing_lab_trace",
    description: "Return the complete in-session decision, write, readback, and rollback trace.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false }
  },
  {
    name: "wing_lab_stop",
    description: "Restore every lab preimage, disconnect, and release the global console writer lock.",
    inputSchema: { type: "object", properties: {}, additionalProperties: false }
  },
  {
    name: "wing_lab_emergency_stop",
    description: "Immediately attempt reverse rollback, disconnect, and close the armed lab session.",
    inputSchema: {
      type: "object",
      properties: { reason: { type: "string", maxLength: 300 } },
      additionalProperties: false
    }
  }
];

const methodByTool = {
  automixer_pipeline_catalog: "pipeline_catalog",
  wing_lab_status: "status",
  wing_lab_arm: "arm",
  wing_lab_apply_decisions: "apply_decisions",
  wing_lab_trace: "trace",
  wing_lab_stop: "stop",
  wing_lab_emergency_stop: "emergency_stop"
};

function writeMessage(message) {
  const body = JSON.stringify(message);
  process.stdout.write(
    `Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`
  );
}

async function handle(message) {
  if (!message || message.jsonrpc !== "2.0") return;
  if (message.method === "notifications/initialized") return;
  try {
    let result;
    if (message.method === "initialize") {
      result = {
        protocolVersion: message.params?.protocolVersion || "2024-11-05",
        capabilities: { tools: {} },
        serverInfo: { name: "automixer-wing-lab", version: "1.0.0" }
      };
    } else if (message.method === "tools/list") {
      result = { tools };
    } else if (message.method === "tools/call") {
      const toolName = message.params?.name;
      const bridgeMethod = methodByTool[toolName];
      if (!bridgeMethod) throw new Error("mcp_tool_not_allowed");
      const payload = await bridgeCall(
        bridgeMethod,
        message.params?.arguments || {}
      );
      result = {
        content: [{ type: "text", text: JSON.stringify(payload) }],
        structuredContent: payload
      };
    } else {
      throw new Error("mcp_method_not_supported");
    }
    if (message.id !== undefined) {
      writeMessage({ jsonrpc: "2.0", id: message.id, result });
    }
  } catch (error) {
    if (message.id !== undefined) {
      writeMessage({
        jsonrpc: "2.0",
        id: message.id,
        error: { code: -32000, message: String(error?.message || error) }
      });
    }
  }
}

process.stdin.on("data", (chunk) => {
  input = Buffer.concat([input, chunk]);
  while (true) {
    const headerEnd = input.indexOf("\r\n\r\n");
    if (headerEnd < 0) break;
    const header = input.subarray(0, headerEnd).toString("utf8");
    const match = /^Content-Length:\s*(\d+)\s*$/im.exec(header);
    if (!match) process.exit(2);
    const length = Number(match[1]);
    const bodyStart = headerEnd + 4;
    if (input.length < bodyStart + length) break;
    const body = input.subarray(bodyStart, bodyStart + length).toString("utf8");
    input = input.subarray(bodyStart + length);
    Promise.resolve(handle(JSON.parse(body))).catch(() => process.exit(3));
  }
});

function shutdown() {
  bridge.kill("SIGTERM");
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
process.stdin.on("end", shutdown);
