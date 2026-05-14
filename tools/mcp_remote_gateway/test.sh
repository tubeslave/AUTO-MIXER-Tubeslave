#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

node --input-type=module <<'NODE'
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { LATEST_PROTOCOL_VERSION } from "@modelcontextprotocol/sdk/types.js";
import { createAutomixerMcpApp, MCP_PATH } from "./server.js";

const app = createAutomixerMcpApp();
const listener = app.listen(0, "127.0.0.1");
const projectRoot = path.resolve(process.cwd(), "../..");

async function readProjectJson(relativePath) {
  return JSON.parse(await fs.readFile(path.resolve(projectRoot, relativePath), "utf8"));
}

async function readProjectText(relativePath) {
  return await fs.readFile(path.resolve(projectRoot, relativePath), "utf8");
}

async function closeListener() {
  await new Promise((resolve, reject) => {
    listener.close((error) => (error ? reject(error) : resolve()));
  });
}

try {
  await new Promise((resolve) => listener.once("listening", resolve));
  const address = listener.address();
  const url = new URL(`http://127.0.0.1:${address.port}${MCP_PATH}`);

  async function httpGetWithHost(pathname, hostHeader) {
    return await new Promise((resolve, reject) => {
      const request = http.request(
        {
          host: "127.0.0.1",
          port: address.port,
          path: pathname,
          method: "GET",
          headers: {
            Host: hostHeader
          }
        },
        (response) => {
          let body = "";
          response.setEncoding("utf8");
          response.on("data", (chunk) => {
            body += chunk;
          });
          response.on("end", () => {
            resolve({ statusCode: response.statusCode, body });
          });
        }
      );
      request.on("error", reject);
      request.end();
    });
  }

  const cloudflareHostCheck = await httpGetWithHost(
    "/health",
    "therapeutic-option-trans-acres.trycloudflare.com"
  );
  assert.equal(cloudflareHostCheck.statusCode, 200);
  assert.equal(JSON.parse(cloudflareHostCheck.body).ok, true);

  const localhostHostCheck = await httpGetWithHost("/health", "localhost");
  assert.equal(localhostHostCheck.statusCode, 200);
  assert.equal(JSON.parse(localhostHostCheck.body).ok, true);

  async function readSseJson(response) {
    const text = await response.text();
    const messages = text
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line))
      .filter((message) => message && Object.keys(message).length > 0);
    assert.ok(messages.length > 0, `Expected at least one SSE data message, got: ${text}`);
    return messages.at(-1);
  }

  const rawInitResponse = await fetch(url, {
    method: "POST",
    headers: {
      Accept: "application/json, text/event-stream",
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: "raw-init",
      method: "initialize",
      params: {
        protocolVersion: LATEST_PROTOCOL_VERSION,
        capabilities: {},
        clientInfo: {
          name: "automixer-raw-validator-test",
          version: "1.0.0"
        }
      }
    })
  });
  assert.equal(rawInitResponse.status, 200);
  assert.match(rawInitResponse.headers.get("content-type") || "", /text\/event-stream/);
  const rawSessionId = rawInitResponse.headers.get("mcp-session-id");
  assert.ok(rawSessionId, "initialize response must include mcp-session-id");
  const rawInit = await readSseJson(rawInitResponse);
  assert.equal(rawInit.id, "raw-init");
  assert.equal(rawInit.result.protocolVersion, LATEST_PROTOCOL_VERSION);
  assert.ok(rawInit.result.capabilities.tools);

  const initializedResponse = await fetch(url, {
    method: "POST",
    headers: {
      Accept: "application/json, text/event-stream",
      "Content-Type": "application/json",
      "Mcp-Session-Id": rawSessionId,
      "MCP-Protocol-Version": LATEST_PROTOCOL_VERSION
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      method: "notifications/initialized"
    })
  });
  assert.equal(initializedResponse.status, 202);

  const rawListResponse = await fetch(url, {
    method: "POST",
    headers: {
      Accept: "application/json, text/event-stream",
      "Content-Type": "application/json",
      "Mcp-Session-Id": rawSessionId,
      "MCP-Protocol-Version": LATEST_PROTOCOL_VERSION
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: "raw-tools-list",
      method: "tools/list",
      params: {}
    })
  });
  assert.equal(rawListResponse.status, 200);
  assert.match(rawListResponse.headers.get("content-type") || "", /text\/event-stream/);
  const rawList = await readSseJson(rawListResponse);
  assert.equal(rawList.id, "raw-tools-list");
  assert.deepEqual(
    rawList.result.tools.map((tool) => tool.name).sort(),
    ["create_task", "git_report", "list_artifacts", "run_agent", "run_director"]
  );
  for (const tool of rawList.result.tools) {
    assert.ok(tool.title, `${tool.name} must include a title`);
    assert.ok(tool.inputSchema, `${tool.name} must include inputSchema`);
    assert.ok(tool.outputSchema, `${tool.name} must include outputSchema`);
    assert.notEqual(tool.annotations?.readOnlyHint, undefined, `${tool.name} must include readOnlyHint`);
    assert.notEqual(tool.annotations?.openWorldHint, undefined, `${tool.name} must include openWorldHint`);
    assert.notEqual(tool.annotations?.destructiveHint, undefined, `${tool.name} must include destructiveHint`);
  }

  const rawSseResponse = await fetch(url, {
    method: "GET",
    headers: {
      Accept: "text/event-stream",
      "Mcp-Session-Id": rawSessionId,
      "MCP-Protocol-Version": LATEST_PROTOCOL_VERSION
    }
  });
  assert.equal(rawSseResponse.status, 200);
  assert.match(rawSseResponse.headers.get("content-type") || "", /text\/event-stream/);
  await rawSseResponse.body.cancel();

  const rawDeleteResponse = await fetch(url, {
    method: "DELETE",
    headers: {
      Accept: "application/json, text/event-stream",
      "Mcp-Session-Id": rawSessionId,
      "MCP-Protocol-Version": LATEST_PROTOCOL_VERSION
    }
  });
  assert.equal(rawDeleteResponse.status, 200);

  const client = new Client({
    name: "automixer-gateway-test",
    version: "1.0.0"
  });
  const transport = new StreamableHTTPClientTransport(url);

  await client.connect(transport);

  const tools = await client.listTools();
  const names = tools.tools.map((tool) => tool.name).sort();
  assert.deepEqual(names, ["create_task", "git_report", "list_artifacts", "run_agent", "run_director"]);

  const taskResult = await client.callTool({
    name: "create_task",
    arguments: {
      title: "MCP gateway smoke test",
      prompt: "Verify create_task writes only a dry-run Paperclip task.",
      dryRun: false,
      metadata: {
        source: "tools/mcp_remote_gateway/test.sh"
      }
    }
  });
  assert.equal(taskResult.structuredContent.ok, true);
  assert.match(taskResult.structuredContent.taskFile, /^\.paperclip\/tasks\/task_/);
  assert.equal(taskResult.structuredContent.dryRun, true);
  assert.equal(taskResult.structuredContent.dryRunEnforced, true);
  assert.equal(taskResult.structuredContent.sourceCodeChangesAllowed, false);

  const taskJson = await readProjectJson(taskResult.structuredContent.taskFile);
  assert.equal(taskJson.dryRun, true);
  assert.equal(taskJson.dryRunEnforced, true);
  assert.equal(taskJson.sourceCodeChangesAllowed, false);
  assert.equal(taskJson.policy.dryRun, true);
  assert.equal(taskJson.policy.dryRunEnforced, true);
  assert.equal(taskJson.policy.sourceCodeChangesAllowed, false);
  assert.deepEqual(taskJson.policy.writesAllowed, [".paperclip/tasks", ".paperclip/reports"]);

  const agentResult = await client.callTool({
    name: "run_agent",
    arguments: {
      agent: "smoke",
      prompt: "Record a dry-run agent request without editing project source code.",
      dryRun: false
    }
  });
  assert.equal(agentResult.structuredContent.ok, true);
  assert.equal(agentResult.structuredContent.dryRun, true);
  assert.equal(agentResult.structuredContent.dryRunEnforced, true);
  assert.equal(agentResult.structuredContent.sourceCodeChangesAllowed, false);
  assert.match(agentResult.structuredContent.taskFile, /^\.paperclip\/tasks\/task_/);
  assert.match(agentResult.structuredContent.reportFile, /^\.paperclip\/reports\//);

  const agentTaskJson = await readProjectJson(agentResult.structuredContent.taskFile);
  assert.equal(agentTaskJson.sourceTool, "run_agent");
  assert.equal(agentTaskJson.dryRun, true);
  assert.equal(agentTaskJson.dryRunEnforced, true);
  assert.equal(agentTaskJson.sourceCodeChangesAllowed, false);
  assert.deepEqual(agentTaskJson.policy.writesAllowed, [".paperclip/tasks", ".paperclip/reports"]);

  const agentReport = await readProjectText(agentResult.structuredContent.reportFile);
  for (const expected of [
    "## Timestamp",
    "## Agent",
    "smoke",
    "## Prompt",
    "## Dry Run Status",
    "dryRun: true",
    "dryRunEnforced: true",
    "## Policy",
    "sourceCodeChangesAllowed: false",
    ".paperclip/tasks, .paperclip/reports",
    "## Task File",
    agentResult.structuredContent.taskFile,
    "## Git Branch",
    "## Git Status",
    "## Next Steps"
  ]) {
    assert.ok(agentReport.includes(expected), `run_agent report must include ${expected}`);
  }

  const directorResult = await client.callTool({
    name: "run_director",
    arguments: {
      title: "MCP gateway director smoke test",
      prompt:
        "Verify run_director creates a dry-run Paperclip task/report, assigns internal roles, and returns the report directly to GPT Chat.",
      mode: "analysis",
      dryRun: false
    }
  });
  assert.equal(directorResult.structuredContent.ok, true);
  assert.match(directorResult.structuredContent.taskFile, /^\.paperclip\/tasks\/task_/);
  assert.match(directorResult.structuredContent.reportFile, /^\.paperclip\/reports\//);
  assert.equal(directorResult.structuredContent.dryRun, true);
  assert.equal(directorResult.structuredContent.dryRunEnforced, true);
  assert.equal(directorResult.structuredContent.sourceCodeChangesAllowed, false);
  assert.equal(typeof directorResult.structuredContent.summary, "string");
  assert.ok(
    directorResult.structuredContent.summary.includes("Paperclip Director completed analysis dry-run"),
    "run_director response must include a useful summary"
  );
  assert.equal(typeof directorResult.structuredContent.directorReport, "string");
  assert.ok(
    directorResult.structuredContent.directorReport.includes("## Role Assignments"),
    "run_director response must contain directorReport"
  );

  const directorTaskJson = await readProjectJson(directorResult.structuredContent.taskFile);
  assert.equal(directorTaskJson.sourceTool, "run_director");
  assert.equal(directorTaskJson.mode, "analysis");
  assert.equal(directorTaskJson.requestedDryRun, false);
  assert.equal(directorTaskJson.dryRun, true);
  assert.equal(directorTaskJson.dryRunEnforced, true);
  assert.equal(directorTaskJson.sourceCodeChangesAllowed, false);
  assert.equal(directorTaskJson.codexAsExecutorAllowed, false);
  assert.equal(directorTaskJson.externalLlmAllowedForInternalRoles, false);
  assert.ok(Array.isArray(directorTaskJson.roleAssignments));
  assert.deepEqual(
    directorTaskJson.roleAssignments.map((assignment) => assignment.role),
    ["Director", "DSP Agent", "ML Agent", "Refactor Agent", "Analyzer Agent", "Safety Agent"]
  );

  const directorReport = await readProjectText(directorResult.structuredContent.reportFile);
  assert.equal(directorReport, directorResult.structuredContent.directorReport);
  for (const expected of [
    "# MCP gateway director smoke test",
    "## Original GPT Prompt",
    "## Director Decision",
    "## Task Breakdown",
    "## Role Assignments",
    "Director",
    "DSP Agent",
    "ML Agent",
    "Refactor Agent",
    "Analyzer Agent",
    "Safety Agent",
    "## Risk Assessment",
    "## Execution Plan",
    "## Next Steps",
    "## Created Task File Path",
    directorResult.structuredContent.taskFile,
    "## Created Report File Path",
    directorResult.structuredContent.reportFile,
    "dryRunEnforced: true",
    "sourceCodeChangesAllowed: false",
    "## Git Branch",
    "## Git Status"
  ]) {
    assert.ok(directorReport.includes(expected), `run_director report must include ${expected}`);
  }

  const gitResult = await client.callTool({
    name: "git_report",
    arguments: {
      title: "MCP gateway smoke git report"
    }
  });
  assert.equal(gitResult.structuredContent.ok, true);
  assert.match(gitResult.structuredContent.reportFile, /^\.paperclip\/reports\//);
  assert.equal(gitResult.structuredContent.dryRun, true);
  assert.equal(gitResult.structuredContent.dryRunEnforced, true);
  assert.equal(gitResult.structuredContent.sourceCodeChangesAllowed, false);

  const artifactResult = await client.callTool({
    name: "list_artifacts",
    arguments: {
      kind: "all",
      limit: 10
    }
  });
  assert.equal(artifactResult.structuredContent.ok, true);
  assert.ok(Array.isArray(artifactResult.structuredContent.artifacts));
  assert.ok(Array.isArray(artifactResult.structuredContent.tasks));
  assert.ok(Array.isArray(artifactResult.structuredContent.reports));
  assert.equal(artifactResult.structuredContent.dryRun, true);
  assert.equal(artifactResult.structuredContent.dryRunEnforced, true);
  assert.equal(artifactResult.structuredContent.sourceCodeChangesAllowed, false);
  assert.ok(
    artifactResult.structuredContent.artifacts.some(
      (artifact) => artifact.file === agentResult.structuredContent.taskFile
    ),
    "list_artifacts must include the run_agent task file"
  );
  assert.ok(
    artifactResult.structuredContent.artifacts.some(
      (artifact) => artifact.file === agentResult.structuredContent.reportFile
    ),
    "list_artifacts must include the run_agent report file"
  );

  await client.close();
  await closeListener();
  console.log("MCP Streamable HTTP smoke test passed.");
} catch (error) {
  await closeListener().catch(() => {});
  throw error;
}
NODE
