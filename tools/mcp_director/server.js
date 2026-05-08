#!/usr/bin/env node
"use strict";

const fs = require("node:fs");
const https = require("node:https");
const os = require("node:os");
const path = require("node:path");
const { randomUUID } = require("node:crypto");
const { spawnSync } = require("node:child_process");

const SERVER_NAME = "automixer-mcp-director";
const SERVER_VERSION = "0.1.0";
const PROTOCOL_VERSION = "2024-11-05";
const DEFAULT_REPO_ROOT = path.resolve(__dirname, "..", "..");
const REPO_ROOT = path.resolve(process.env.MCP_DIRECTOR_REPO || DEFAULT_REPO_ROOT);
const PAPERCLIP_ROOT = path.join(REPO_ROOT, ".paperclip");
const TASK_DIR = path.join(PAPERCLIP_ROOT, "tasks");
const REPORT_DIR = path.join(PAPERCLIP_ROOT, "reports");
const MAX_AGENT_OUTPUT_CHARS = 12000;
const TELEGRAM_CHUNK_LIMIT = 3900;
const PROTECTED_BRANCH_RE = /^(main|master|production|prod|release)$/i;

const AGENTS = {
  dsp: {
    label: "DSP Agent",
    focus: "audio DSP, mix pipeline math, LUFS/true-peak behavior, routing, WING/live-safety constraints"
  },
  ml: {
    label: "ML Agent",
    focus: "ML evaluation, model/data assumptions, inference boundaries, metrics, reproducibility"
  },
  refactor: {
    label: "Refactor Agent",
    focus: "code structure, maintainability, interfaces, isolated opt-in changes, test seams"
  },
  analyzer: {
    label: "Analyzer Agent",
    focus: "repo analysis, risk inventory, tests, reports, operational readiness"
  }
};

const TOOLS = [
  {
    name: "create_task",
    description: "Create a Paperclip task JSON under .paperclip/tasks.",
    inputSchema: {
      type: "object",
      properties: {
        title: { type: "string", description: "Short task title." },
        description: { type: "string", description: "Detailed task instructions." },
        agent: {
          type: "string",
          enum: ["dsp", "ml", "refactor", "analyzer", "director"],
          description: "Preferred agent role."
        },
        priority: {
          type: "string",
          enum: ["low", "normal", "high"],
          description: "Task priority."
        },
        targetBranch: { type: "string", description: "Suggested git branch for later implementation." },
        metadata: { type: "object", description: "Optional structured metadata." },
        dryRun: {
          type: "boolean",
          description: "Preview the task without writing a JSON file. Defaults to false for this tool."
        }
      },
      required: ["title"],
      additionalProperties: false
    }
  },
  {
    name: "run_agent",
    description: "Create a task and optionally run a read-only Codex agent, writing the result to .paperclip/reports.",
    inputSchema: {
      type: "object",
      properties: {
        agent: {
          type: "string",
          enum: ["dsp", "ml", "refactor", "analyzer"],
          description: "Agent role to run."
        },
        title: { type: "string", description: "Task title." },
        prompt: { type: "string", description: "Agent instructions." },
        taskId: { type: "string", description: "Existing .paperclip task id or filename." },
        scope: { type: "string", description: "Repo area, file, feature, or pipeline to inspect." },
        targetBranch: { type: "string", description: "Suggested implementation branch." },
        timeoutSeconds: {
          type: "number",
          description: "Codex execution timeout when dryRun is false. Defaults to 300."
        },
        dryRun: {
          type: "boolean",
          description: "When true, do not invoke Codex CLI. Defaults to true."
        }
      },
      required: ["agent"],
      additionalProperties: false
    }
  },
  {
    name: "run_full_review",
    description: "Plan or run DSP, ML, Refactor, and Analyzer read-only review agents; write review report markdown.",
    inputSchema: {
      type: "object",
      properties: {
        scope: { type: "string", description: "Review scope. Defaults to full repository." },
        agents: {
          type: "array",
          items: { type: "string", enum: ["dsp", "ml", "refactor", "analyzer"] },
          description: "Subset of agents. Defaults to all agents."
        },
        title: { type: "string", description: "Review title." },
        timeoutSeconds: {
          type: "number",
          description: "Per-agent Codex timeout when dryRun is false. Defaults to 300."
        },
        dryRun: {
          type: "boolean",
          description: "When true, create tasks and a manual review plan only. Defaults to true."
        }
      },
      additionalProperties: false
    }
  },
  {
    name: "git_report",
    description: "Write a read-only git status/branch report to .paperclip/reports.",
    inputSchema: {
      type: "object",
      properties: {
        includeDiff: {
          type: "boolean",
          description: "Include truncated git diff output. Defaults to false."
        },
        maxDiffLines: {
          type: "number",
          description: "Maximum diff lines if includeDiff is true. Defaults to 300."
        },
        title: { type: "string", description: "Report title." }
      },
      additionalProperties: false
    }
  },
  {
    name: "telegram_notify",
    description: "Send or dry-run a Telegram notification and write a notification report markdown.",
    inputSchema: {
      type: "object",
      properties: {
        message: { type: "string", description: "Telegram message text." },
        reportPath: { type: "string", description: "Optional report path under the repository to append." },
        dryRun: {
          type: "boolean",
          description: "When true, do not call Telegram API. Defaults to true."
        },
        tokenEnv: {
          type: "string",
          description: "Environment variable containing bot token. Defaults to TELEGRAM_BOT_TOKEN."
        },
        chatId: { type: "string", description: "Telegram chat id. Defaults to TELEGRAM_CHAT_ID env." },
        chatIdEnv: {
          type: "string",
          description: "Environment variable containing chat id. Defaults to TELEGRAM_CHAT_ID."
        }
      },
      required: ["message"],
      additionalProperties: false
    }
  }
];

let inputBuffer = Buffer.alloc(0);

process.stdin.on("data", (chunk) => {
  inputBuffer = Buffer.concat([inputBuffer, chunk]);
  drainInput().catch((error) => {
    logError(error);
  });
});

process.stdin.on("end", () => {
  process.exit(0);
});

process.on("uncaughtException", (error) => {
  logError(error);
});

function drainInput() {
  return new Promise((resolve) => {
    while (true) {
      const parsed = readFrame(inputBuffer);
      if (!parsed) {
        break;
      }
      inputBuffer = inputBuffer.subarray(parsed.nextOffset);
      handleMessage(parsed.message).catch((error) => {
        logError(error);
      });
    }
    resolve();
  });
}

async function handleMessage(message) {
  if (!message || message.jsonrpc !== "2.0") {
    return;
  }

  const hasId = Object.prototype.hasOwnProperty.call(message, "id");
  if (!hasId) {
    return;
  }

  try {
    const result = await handleRequest(message.method, message.params || {});
    sendFrame({ jsonrpc: "2.0", id: message.id, result });
  } catch (error) {
    sendFrame({
      jsonrpc: "2.0",
      id: message.id,
      error: {
        code: error.code || -32603,
        message: error.message || String(error)
      }
    });
  }
}

async function handleRequest(method, params) {
  switch (method) {
    case "initialize":
      return {
        protocolVersion: params.protocolVersion || PROTOCOL_VERSION,
        capabilities: {
          tools: {}
        },
        serverInfo: {
          name: SERVER_NAME,
          version: SERVER_VERSION
        },
        instructions:
          "Local Paperclip Director for AUTO-MIXER. Tasks are stored in .paperclip/tasks and reports in .paperclip/reports. External side effects default to dry-run."
      };
    case "ping":
      return {};
    case "tools/list":
      return { tools: TOOLS };
    case "tools/call":
      return callTool(params.name, params.arguments || {});
    case "resources/list":
      return { resources: [] };
    case "prompts/list":
      return { prompts: [] };
    default: {
      const error = new Error(`Unsupported MCP method: ${method}`);
      error.code = -32601;
      throw error;
    }
  }
}

async function callTool(name, args) {
  try {
    let result;
    switch (name) {
      case "create_task":
        result = createTask(args);
        break;
      case "run_agent":
        result = runAgent(args);
        break;
      case "run_full_review":
        result = runFullReview(args);
        break;
      case "git_report":
        result = gitReport(args);
        break;
      case "telegram_notify":
        result = await telegramNotify(args);
        break;
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
    return toolResult(result);
  } catch (error) {
    return toolResult(
      {
        ok: false,
        error: error.message || String(error)
      },
      true
    );
  }
}

function createTask(args) {
  ensureRepo();
  const title = requireString(args.title, "title");
  const dryRun = Boolean(args.dryRun);
  const branch = currentBranch();
  const agent = normalizeAgent(args.agent || "analyzer", true);
  const task = buildTaskRecord({
    title,
    description: optionalString(args.description),
    agent,
    priority: args.priority || "normal",
    targetBranch: optionalString(args.targetBranch),
    metadata: args.metadata || {},
    currentBranch: branch
  });

  if (dryRun) {
    return {
      ok: true,
      dryRun: true,
      branch,
      task,
      note: "Preview only; no task file was written."
    };
  }

  assertWritableBranch(branch);
  ensurePaperclipDirs();
  const filePath = taskPath(task);
  writeJsonAtomic(filePath, task);
  return {
    ok: true,
    dryRun: false,
    branch,
    taskId: task.id,
    taskPath: relativeToRepo(filePath),
    task
  };
}

function runAgent(args) {
  ensureRepo();
  const agent = normalizeAgent(args.agent);
  const dryRun = args.dryRun !== false;
  const branch = currentBranch();
  assertWritableBranch(branch);
  ensurePaperclipDirs();

  const existingTask = args.taskId ? readTask(args.taskId) : null;
  const title =
    optionalString(args.title) ||
    (existingTask && existingTask.title) ||
    `${AGENTS[agent].label}: ${optionalString(args.scope) || "repository task"}`;
  const prompt =
    optionalString(args.prompt) ||
    (existingTask && existingTask.description) ||
    optionalString(args.scope) ||
    "";
  if (!prompt) {
    throw new Error("run_agent requires prompt, scope, or an existing taskId with description.");
  }

  const task =
    existingTask ||
    buildTaskRecord({
      title,
      description: prompt,
      agent,
      priority: "normal",
      targetBranch: optionalString(args.targetBranch),
      metadata: { scope: optionalString(args.scope) },
      currentBranch: branch
    });
  if (!existingTask) {
    writeJsonAtomic(taskPath(task), task);
  }

  const reportBase = `${timestamp()}_${safeSlug(agent)}_${safeSlug(title)}`;
  const reportPath = path.join(REPORT_DIR, `${reportBase}.md`);
  const timeoutSeconds = clampNumber(args.timeoutSeconds, 300, 30, 1800);

  if (dryRun) {
    const report = manualAgentMarkdown({
      agent,
      task,
      prompt,
      branch,
      reason: "dryRun=true; Codex CLI was not invoked.",
      timeoutSeconds
    });
    writeTextAtomic(reportPath, report);
    updateTaskStatus(task, existingTask, "planned", { reportPath: relativeToRepo(reportPath), dryRun: true });
    return {
      ok: true,
      dryRun: true,
      branch,
      taskId: task.id,
      taskPath: relativeToRepo(taskPath(task)),
      reportPath: relativeToRepo(reportPath),
      note: "Created manual agent task/report; Codex CLI was not invoked."
    };
  }

  if (!commandExists("codex")) {
    const report = manualAgentMarkdown({
      agent,
      task,
      prompt,
      branch,
      reason: "Codex CLI was not found in PATH.",
      timeoutSeconds
    });
    writeTextAtomic(reportPath, report);
    updateTaskStatus(task, existingTask, "manual_fallback", {
      reportPath: relativeToRepo(reportPath),
      dryRun: false,
      fallback: "codex_cli_missing"
    });
    return {
      ok: true,
      dryRun: false,
      fallback: "codex_cli_missing",
      branch,
      taskId: task.id,
      taskPath: relativeToRepo(taskPath(task)),
      reportPath: relativeToRepo(reportPath)
    };
  }

  const tmpOutput = path.join(os.tmpdir(), `${SERVER_NAME}-${task.id}.txt`);
  const codexPrompt = buildCodexPrompt({ agent, task, prompt, branch });
  const completed = spawnSync(
    "codex",
    ["exec", "-C", REPO_ROOT, "-s", "read-only", "--output-last-message", tmpOutput, codexPrompt],
    {
      cwd: REPO_ROOT,
      encoding: "utf8",
      timeout: timeoutSeconds * 1000,
      maxBuffer: 1024 * 1024 * 10
    }
  );

  const finalMessage = safeReadText(tmpOutput).trim();
  safeUnlink(tmpOutput);
  const outputText = truncateText(
    finalMessage || completed.stdout || completed.stderr || "Codex CLI returned no text.",
    MAX_AGENT_OUTPUT_CHARS
  );
  const report = agentReportMarkdown({
    agent,
    task,
    branch,
    dryRun,
    exitCode: exitCodeOf(completed),
    timedOut: Boolean(completed.error && completed.error.code === "ETIMEDOUT"),
    outputText
  });
  writeTextAtomic(reportPath, report);
  updateTaskStatus(task, existingTask, completed.status === 0 ? "reported" : "codex_error", {
    reportPath: relativeToRepo(reportPath),
    dryRun: false,
    exitCode: exitCodeOf(completed)
  });

  return {
    ok: completed.status === 0,
    dryRun: false,
    branch,
    taskId: task.id,
    taskPath: relativeToRepo(taskPath(task)),
    reportPath: relativeToRepo(reportPath),
    codexExitCode: exitCodeOf(completed)
  };
}

function runFullReview(args) {
  ensureRepo();
  const dryRun = args.dryRun !== false;
  const branch = currentBranch();
  assertWritableBranch(branch);
  ensurePaperclipDirs();

  const agents = normalizeAgents(args.agents);
  const scope = optionalString(args.scope) || "full repository";
  const title = optionalString(args.title) || `AUTO-MIXER full review: ${scope}`;
  const timeoutSeconds = clampNumber(args.timeoutSeconds, 300, 30, 1800);
  const taskResults = [];
  const sections = [
    `# ${title}`,
    "",
    `- Created: ${new Date().toISOString()}`,
    `- Repository: ${REPO_ROOT}`,
    `- Branch: ${branch || "(detached HEAD)"}`,
    `- Scope: ${scope}`,
    `- Dry run: ${dryRun}`,
    "",
    "## Agent Results"
  ];

  for (const agent of agents) {
    const result = runAgent({
      agent,
      title: `${AGENTS[agent].label} review: ${scope}`,
      prompt: `Review scope: ${scope}`,
      scope,
      timeoutSeconds,
      dryRun
    });
    taskResults.push(result);
    sections.push(
      "",
      `### ${AGENTS[agent].label}`,
      "",
      `- Task: ${result.taskPath}`,
      `- Report: ${result.reportPath}`,
      `- Status: ${result.ok ? "ok" : "needs attention"}`,
      `- Dry run: ${result.dryRun}`
    );
  }

  const reportPath = path.join(REPORT_DIR, `${timestamp()}_full_review_${safeSlug(scope)}.md`);
  sections.push(
    "",
    "## Safety Notes",
    "",
    "- Director writes only .paperclip task/report artifacts.",
    "- Codex agent execution uses read-only sandbox.",
    "- Implementation branches must be created/applied manually after reviewing reports."
  );
  writeTextAtomic(reportPath, `${sections.join("\n")}\n`);

  return {
    ok: taskResults.every((result) => result.ok),
    dryRun,
    branch,
    agents,
    reportPath: relativeToRepo(reportPath),
    tasks: taskResults.map((result) => ({
      taskId: result.taskId,
      taskPath: result.taskPath,
      reportPath: result.reportPath,
      ok: result.ok
    }))
  };
}

function gitReport(args) {
  ensureRepo();
  const branch = currentBranch();
  assertWritableBranch(branch);
  ensurePaperclipDirs();

  const includeDiff = Boolean(args.includeDiff);
  const maxDiffLines = clampNumber(args.maxDiffLines, 300, 20, 2000);
  const title = optionalString(args.title) || "AUTO-MIXER Git Report";
  const status = runGit(["status", "--short", "--branch"]);
  const log = runGit(["log", "--oneline", "-8"]);
  const stat = runGit(["diff", "--stat"]);
  const stagedStat = runGit(["diff", "--cached", "--stat"]);
  const changedFiles = runGit(["diff", "--name-status"]);
  const stagedFiles = runGit(["diff", "--cached", "--name-status"]);
  const untracked = runGit(["ls-files", "--others", "--exclude-standard"]);

  const sections = [
    `# ${title}`,
    "",
    `- Created: ${new Date().toISOString()}`,
    `- Repository: ${REPO_ROOT}`,
    `- Branch: ${branch || "(detached HEAD)"}`,
    "",
    "## Status",
    "",
    fenced(status.output || "(clean)"),
    "",
    "## Recent Commits",
    "",
    fenced(log.output || log.error || "(no commits)"),
    "",
    "## Diff Stat",
    "",
    fenced(stat.output || "(no unstaged diff)"),
    "",
    "## Staged Diff Stat",
    "",
    fenced(stagedStat.output || "(no staged diff)"),
    "",
    "## Changed Files",
    "",
    fenced(changedFiles.output || "(none)"),
    "",
    "## Staged Files",
    "",
    fenced(stagedFiles.output || "(none)"),
    "",
    "## Untracked Files",
    "",
    fenced(untracked.output || "(none)")
  ];

  if (includeDiff) {
    const diff = runGit(["diff"]);
    sections.push("", `## Diff (first ${maxDiffLines} lines)`, "", fenced(firstLines(diff.output || diff.error || "", maxDiffLines)));
  }

  const reportPath = path.join(REPORT_DIR, `${timestamp()}_git_report.md`);
  writeTextAtomic(reportPath, `${sections.join("\n")}\n`);
  return {
    ok: true,
    branch,
    reportPath: relativeToRepo(reportPath),
    includeDiff
  };
}

async function telegramNotify(args) {
  ensureRepo();
  const branch = currentBranch();
  assertWritableBranch(branch);
  ensurePaperclipDirs();

  const message = requireString(args.message, "message");
  const dryRun = args.dryRun !== false;
  const tokenEnv = optionalString(args.tokenEnv) || "TELEGRAM_BOT_TOKEN";
  const chatIdEnv = optionalString(args.chatIdEnv) || "TELEGRAM_CHAT_ID";
  const token = process.env[tokenEnv];
  const chatId = optionalString(args.chatId) || process.env[chatIdEnv] || process.env.TELEGRAM_ALLOWED_CHAT_ID;
  const reportBody = args.reportPath ? safeReadRepoText(args.reportPath) : "";
  const fullMessage = [message, reportBody ? `\n\n${reportBody}` : ""].join("").trim();
  const reportPath = path.join(REPORT_DIR, `${timestamp()}_telegram_notify.md`);

  if (dryRun) {
    writeTextAtomic(
      reportPath,
      notificationReportMarkdown({
        branch,
        dryRun,
        delivered: false,
        reason: "dryRun=true; Telegram API was not called.",
        message: fullMessage
      })
    );
    return {
      ok: true,
      dryRun: true,
      branch,
      reportPath: relativeToRepo(reportPath),
      note: "Telegram notification dry-run report written."
    };
  }

  if (!token || !chatId) {
    writeTextAtomic(
      reportPath,
      notificationReportMarkdown({
        branch,
        dryRun,
        delivered: false,
        reason: `Missing ${token ? "" : tokenEnv}${!token && !chatId ? " and " : ""}${chatId ? "" : chatIdEnv}.`,
        message: fullMessage
      })
    );
    return {
      ok: true,
      dryRun: false,
      fallback: "missing_telegram_env",
      branch,
      reportPath: relativeToRepo(reportPath)
    };
  }

  const chunks = chunkText(fullMessage, TELEGRAM_CHUNK_LIMIT);
  for (const chunk of chunks) {
    await telegramRequest(token, "sendMessage", { chat_id: chatId, text: chunk });
  }
  writeTextAtomic(
    reportPath,
    notificationReportMarkdown({
      branch,
      dryRun,
      delivered: true,
      reason: `Delivered ${chunks.length} Telegram message chunk(s).`,
      message: fullMessage
    })
  );
  return {
    ok: true,
    dryRun: false,
    delivered: true,
    chunks: chunks.length,
    branch,
    reportPath: relativeToRepo(reportPath)
  };
}

function buildTaskRecord({ title, description, agent, priority, targetBranch, metadata, currentBranch }) {
  const now = new Date().toISOString();
  const id = `task_${timestamp()}_${randomUUID().slice(0, 8)}`;
  return {
    id,
    title,
    description: description || "",
    agent,
    priority,
    status: "queued",
    repoRoot: REPO_ROOT,
    currentBranch: currentBranch || "(detached HEAD)",
    targetBranch: targetBranch || "",
    createdAt: now,
    updatedAt: now,
    metadata: metadata && typeof metadata === "object" && !Array.isArray(metadata) ? metadata : {}
  };
}

function updateTaskStatus(task, existed, status, extra) {
  const updated = {
    ...task,
    status,
    updatedAt: new Date().toISOString(),
    result: {
      ...(task.result || {}),
      ...extra
    }
  };
  const filePath = taskPath(task);
  writeJsonAtomic(filePath, updated);
  if (existed) {
    Object.assign(existed, updated);
  }
}

function taskPath(task) {
  return path.join(TASK_DIR, `${task.id}_${safeSlug(task.title)}.json`);
}

function readTask(taskId) {
  ensurePaperclipDirs();
  const safeId = path.basename(taskId);
  const candidates = fs.readdirSync(TASK_DIR).filter((name) => name === safeId || name.startsWith(`${safeId}_`) || name.startsWith(safeId));
  if (candidates.length === 0) {
    throw new Error(`Task not found: ${taskId}`);
  }
  const filePath = path.join(TASK_DIR, candidates[0]);
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function buildCodexPrompt({ agent, task, prompt, branch }) {
  return [
    `You are ${AGENTS[agent].label} for AUTO-MIXER-Tubeslave.`,
    "",
    "Work in read-only mode. Do not edit files. Do not run destructive commands.",
    "Return a concise Markdown report with findings, risks, concrete file references, suggested tests, and a safe branch plan.",
    "Any command that would mutate repo state must be written as a proposed command and clearly marked as requiring a manual dry-run first.",
    "",
    `Repository: ${REPO_ROOT}`,
    `Current branch: ${branch || "(detached HEAD)"}`,
    `Task id: ${task.id}`,
    `Task title: ${task.title}`,
    `Agent focus: ${AGENTS[agent].focus}`,
    "",
    "User task:",
    prompt
  ].join("\n");
}

function manualAgentMarkdown({ agent, task, prompt, branch, reason, timeoutSeconds }) {
  const command = [
    "codex",
    "exec",
    "-C",
    shellQuote(REPO_ROOT),
    "-s",
    "read-only",
    shellQuote(buildCodexPrompt({ agent, task, prompt, branch }))
  ].join(" ");
  return [
    `# Manual ${AGENTS[agent].label} Task`,
    "",
    `- Created: ${new Date().toISOString()}`,
    `- Reason: ${reason}`,
    `- Repository: ${REPO_ROOT}`,
    `- Branch: ${branch || "(detached HEAD)"}`,
    `- Task: ${task.id}`,
    `- Timeout seconds: ${timeoutSeconds}`,
    "",
    "## Prompt",
    "",
    prompt,
    "",
    "## Manual Command",
    "",
    "Run this from a terminal after reviewing branch state:",
    "",
    fenced(command, "bash"),
    "",
    "## Safety",
    "",
    "- The command uses Codex read-only sandbox.",
    "- Apply any generated implementation in a separate feature branch only after manual review.",
    "- Do not run destructive git or filesystem commands unless they are first reviewed as dry-run proposals."
  ].join("\n") + "\n";
}

function agentReportMarkdown({ agent, task, branch, dryRun, exitCode, timedOut, outputText }) {
  return [
    `# ${AGENTS[agent].label} Report`,
    "",
    `- Created: ${new Date().toISOString()}`,
    `- Repository: ${REPO_ROOT}`,
    `- Branch: ${branch || "(detached HEAD)"}`,
    `- Task: ${task.id}`,
    `- Dry run: ${dryRun}`,
    `- Codex exit code: ${exitCode}`,
    `- Timed out: ${timedOut}`,
    "",
    "## Result",
    "",
    outputText
  ].join("\n") + "\n";
}

function notificationReportMarkdown({ branch, dryRun, delivered, reason, message }) {
  return [
    "# Telegram Notification",
    "",
    `- Created: ${new Date().toISOString()}`,
    `- Repository: ${REPO_ROOT}`,
    `- Branch: ${branch || "(detached HEAD)"}`,
    `- Dry run: ${dryRun}`,
    `- Delivered: ${delivered}`,
    `- Result: ${reason}`,
    "",
    "## Message",
    "",
    message
  ].join("\n") + "\n";
}

function toolResult(result, isError = false) {
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(result, null, 2)
      }
    ],
    isError
  };
}

function readFrame(buffer) {
  const headerEnd = findHeaderEnd(buffer);
  if (headerEnd < 0) {
    return null;
  }
  const header = buffer.subarray(0, headerEnd.index).toString("utf8");
  const match = /^Content-Length:\s*(\d+)\s*$/im.exec(header);
  if (!match) {
    throw new Error("Missing Content-Length header in MCP frame.");
  }
  const contentLength = Number(match[1]);
  const bodyStart = headerEnd.index + headerEnd.length;
  const bodyEnd = bodyStart + contentLength;
  if (buffer.length < bodyEnd) {
    return null;
  }
  const body = buffer.subarray(bodyStart, bodyEnd).toString("utf8");
  return {
    message: JSON.parse(body),
    nextOffset: bodyEnd
  };
}

function findHeaderEnd(buffer) {
  const crlf = buffer.indexOf("\r\n\r\n");
  if (crlf >= 0) {
    return { index: crlf, length: 4 };
  }
  const lf = buffer.indexOf("\n\n");
  if (lf >= 0) {
    return { index: lf, length: 2 };
  }
  return -1;
}

function sendFrame(message) {
  const body = JSON.stringify(message);
  process.stdout.write(`Content-Length: ${Buffer.byteLength(body, "utf8")}\r\n\r\n${body}`);
}

function ensureRepo() {
  if (!fs.existsSync(REPO_ROOT)) {
    throw new Error(`Repository root does not exist: ${REPO_ROOT}`);
  }
  if (!fs.existsSync(path.join(REPO_ROOT, ".git"))) {
    throw new Error(`Repository root is not a git checkout: ${REPO_ROOT}`);
  }
}

function ensurePaperclipDirs() {
  fs.mkdirSync(TASK_DIR, { recursive: true });
  fs.mkdirSync(REPORT_DIR, { recursive: true });
}

function assertWritableBranch(branch) {
  if (!branch) {
    throw new Error("Refusing to write Paperclip artifacts on detached HEAD.");
  }
  const requiredBranch = process.env.MCP_DIRECTOR_REQUIRED_BRANCH;
  if (requiredBranch && branch !== requiredBranch) {
    throw new Error(`Refusing to write Paperclip artifacts on ${branch}; MCP_DIRECTOR_REQUIRED_BRANCH=${requiredBranch}.`);
  }
  if (PROTECTED_BRANCH_RE.test(branch)) {
    throw new Error(`Refusing to write Paperclip artifacts on protected branch: ${branch}. Switch to a feature branch first.`);
  }
}

function currentBranch() {
  const result = runGit(["branch", "--show-current"]);
  return result.ok ? result.output.trim() : "";
}

function runGit(args) {
  const completed = spawnSync("git", args, {
    cwd: REPO_ROOT,
    encoding: "utf8",
    timeout: 30000,
    maxBuffer: 1024 * 1024 * 10
  });
  return {
    ok: completed.status === 0,
    code: exitCodeOf(completed),
    output: (completed.stdout || "").trim(),
    error: (completed.stderr || completed.error?.message || "").trim()
  };
}

function commandExists(command) {
  const completed = spawnSync("which", [command], {
    encoding: "utf8",
    timeout: 5000
  });
  return completed.status === 0;
}

function telegramRequest(token, method, payload) {
  const body = JSON.stringify(payload);
  const options = {
    hostname: "api.telegram.org",
    method: "POST",
    path: `/bot${token}/${method}`,
    headers: {
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(body)
    },
    timeout: 20000
  };

  return new Promise((resolve, reject) => {
    const request = https.request(options, (response) => {
      const chunks = [];
      response.on("data", (chunk) => chunks.push(chunk));
      response.on("end", () => {
        const text = Buffer.concat(chunks).toString("utf8");
        let parsed;
        try {
          parsed = JSON.parse(text);
        } catch (error) {
          reject(new Error(`Telegram returned invalid JSON: ${text}`));
          return;
        }
        if (!parsed.ok) {
          reject(new Error(`Telegram API error: ${text}`));
          return;
        }
        resolve(parsed);
      });
    });
    request.on("timeout", () => {
      request.destroy(new Error("Telegram request timed out."));
    });
    request.on("error", reject);
    request.write(body);
    request.end();
  });
}

function normalizeAgent(raw, allowDirector = false) {
  const value = String(raw || "").trim().toLowerCase().replace(/\s+agent$/, "");
  if (allowDirector && value === "director") {
    return "director";
  }
  if (!Object.prototype.hasOwnProperty.call(AGENTS, value)) {
    throw new Error(`Unknown agent "${raw}". Use one of: ${Object.keys(AGENTS).join(", ")}.`);
  }
  return value;
}

function normalizeAgents(rawAgents) {
  if (!rawAgents || rawAgents.length === 0) {
    return Object.keys(AGENTS);
  }
  if (!Array.isArray(rawAgents)) {
    throw new Error("agents must be an array.");
  }
  return rawAgents.map((agent) => normalizeAgent(agent));
}

function requireString(value, name) {
  const stringValue = optionalString(value);
  if (!stringValue) {
    throw new Error(`${name} is required.`);
  }
  return stringValue;
}

function optionalString(value) {
  return typeof value === "string" ? value.trim() : "";
}

function clampNumber(value, defaultValue, min, max) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return defaultValue;
  }
  return Math.max(min, Math.min(max, Math.floor(number)));
}

function timestamp() {
  return new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function safeSlug(value) {
  const slug = String(value || "")
    .normalize("NFKD")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 72);
  return slug || "task";
}

function writeJsonAtomic(filePath, data) {
  writeTextAtomic(filePath, `${JSON.stringify(data, null, 2)}\n`);
}

function writeTextAtomic(filePath, text) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const tmp = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  fs.writeFileSync(tmp, text, "utf8");
  fs.renameSync(tmp, filePath);
}

function safeReadText(filePath) {
  try {
    return fs.readFileSync(filePath, "utf8");
  } catch {
    return "";
  }
}

function safeReadRepoText(filePath) {
  const absolute = path.resolve(REPO_ROOT, filePath);
  if (!isUnder(absolute, REPO_ROOT)) {
    throw new Error(`reportPath must stay under repository root: ${filePath}`);
  }
  return safeReadText(absolute);
}

function safeUnlink(filePath) {
  try {
    fs.unlinkSync(filePath);
  } catch {
    // Ignore cleanup failures for temporary Codex output.
  }
}

function isUnder(child, parent) {
  const relative = path.relative(parent, child);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function relativeToRepo(filePath) {
  return path.relative(REPO_ROOT, filePath).split(path.sep).join("/");
}

function fenced(text, language = "") {
  return `\`\`\`${language}\n${text || ""}\n\`\`\``;
}

function firstLines(text, maxLines) {
  return String(text || "")
    .split(/\r?\n/)
    .slice(0, maxLines)
    .join("\n");
}

function truncateText(text, maxChars) {
  const value = String(text || "");
  if (value.length <= maxChars) {
    return value;
  }
  return `${value.slice(0, maxChars)}\n\n[truncated at ${maxChars} characters]`;
}

function chunkText(text, size) {
  const value = String(text || "").trim() || "(empty)";
  const chunks = [];
  for (let i = 0; i < value.length; i += size) {
    chunks.push(value.slice(i, i + size));
  }
  return chunks;
}

function shellQuote(value) {
  return `'${String(value).replace(/'/g, "'\\''")}'`;
}

function exitCodeOf(completed) {
  if (typeof completed.status === "number") {
    return completed.status;
  }
  if (completed.error) {
    return completed.error.code || 1;
  }
  return 1;
}

function logError(error) {
  process.stderr.write(`[${SERVER_NAME}] ${error.stack || error.message || String(error)}\n`);
}
