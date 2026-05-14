import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { randomUUID } from "node:crypto";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { createMcpExpressApp } from "@modelcontextprotocol/sdk/server/express.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";
import { InMemoryEventStore } from "@modelcontextprotocol/sdk/examples/shared/inMemoryEventStore.js";
import { z } from "zod/v4";

export const PORT = 8787;
export const MCP_PATH = "/mcp/automixer-ab31a16f07c6b";
export const PUBLIC_MCP_URL =
  "https://angela-passport-flexibility-marketplace.trycloudflare.com/mcp/automixer-ab31a16f07c6b";

const execFileAsync = promisify(execFile);
const gatewayDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(gatewayDir, "../..");
const paperclipDir = path.join(projectRoot, ".paperclip");
const tasksDir = path.join(paperclipDir, "tasks");
const reportsDir = path.join(paperclipDir, "reports");
const allowedWriteDirs = [tasksDir, reportsDir].map((dir) => path.resolve(dir));
const allowedWriteRelDirs = [".paperclip/tasks", ".paperclip/reports"];
const allowedGitCommands = new Set([
  "branch\u0000--show-current",
  "status\u0000--short\u0000--branch",
  "rev-parse\u0000--show-toplevel",
  "log\u0000-1\u0000--oneline"
]);
const gatewayPolicy = Object.freeze({
  dryRun: true,
  dryRunEnforced: true,
  writesAllowed: allowedWriteRelDirs,
  sourceCodeChangesAllowed: false,
  shellCommandsAllowed: [
    "git branch --show-current",
    "git status --short --branch",
    "git rev-parse --show-toplevel",
    "git log -1 --oneline"
  ],
  prohibitedActions: [
    "source code changes",
    "dangerous shell commands",
    "git commit",
    "git checkout"
  ]
});

const defaultAllowedHosts = [
  "127.0.0.1",
  "localhost",
  "[::1]",
  "angela-passport-flexibility-marketplace.trycloudflare.com",
  "pontiac-streaming-desktops-political.trycloudflare.com"
];

const createTaskInput = {
  title: z.string().trim().min(1).max(160).default("AUTO-MIXER task"),
  prompt: z.string().trim().min(1).max(20000),
  agent: z.string().trim().min(1).max(80).default("automixer"),
  dryRun: z.boolean().default(true),
  metadata: z.record(z.string(), z.unknown()).optional()
};

const runAgentInput = {
  prompt: z.string().trim().min(1).max(20000),
  agent: z.string().trim().min(1).max(80).default("automixer"),
  dryRun: z.boolean().default(true),
  metadata: z.record(z.string(), z.unknown()).optional()
};

const runDirectorInput = {
  title: z.string().trim().min(1).max(160),
  prompt: z.string().trim().min(1).max(20000),
  mode: z.enum(["analysis", "planning", "implementation_plan"]).default("analysis"),
  dryRun: z.boolean().default(true)
};

const gitReportInput = {
  includeLog: z.boolean().default(true),
  dryRun: z.boolean().default(true),
  title: z.string().trim().min(1).max(160).default("AUTO-MIXER git report")
};

const listArtifactsInput = {
  kind: z.enum(["tasks", "reports", "all"]).default("all"),
  limit: z.number().int().min(1).max(100).default(25)
};

const artifactOutput = z.object({
  kind: z.enum(["tasks", "reports"]),
  file: z.string(),
  sizeBytes: z.number(),
  modifiedAt: z.string()
});

const taskOutput = {
  ok: z.boolean(),
  taskFile: z.string(),
  taskId: z.string(),
  dryRun: z.boolean(),
  dryRunEnforced: z.boolean(),
  sourceCodeChangesAllowed: z.boolean()
};

const runAgentOutput = {
  ok: z.boolean(),
  taskFile: z.string(),
  reportFile: z.string(),
  dryRun: z.boolean(),
  dryRunEnforced: z.boolean(),
  sourceCodeChangesAllowed: z.boolean()
};

const runDirectorOutput = {
  ok: z.boolean(),
  taskFile: z.string(),
  reportFile: z.string(),
  summary: z.string(),
  directorReport: z.string(),
  dryRun: z.boolean(),
  dryRunEnforced: z.boolean(),
  sourceCodeChangesAllowed: z.boolean()
};

const gitSnapshotOutput = z.object({
  command: z.string(),
  stdout: z.string(),
  stderr: z.string()
});

const gitReportOutput = {
  ok: z.boolean(),
  reportFile: z.string(),
  dryRun: z.boolean(),
  dryRunEnforced: z.boolean(),
  sourceCodeChangesAllowed: z.boolean(),
  snapshot: z.array(gitSnapshotOutput)
};

const listArtifactsOutput = {
  ok: z.boolean(),
  artifacts: z.array(artifactOutput),
  tasks: z.array(artifactOutput),
  reports: z.array(artifactOutput),
  count: z.number(),
  dryRun: z.boolean(),
  dryRunEnforced: z.boolean(),
  sourceCodeChangesAllowed: z.boolean()
};

function parseCsvList(value, defaults) {
  const result = new Set(defaults);
  for (const item of (value || "").split(",")) {
    const trimmed = item.trim();
    if (trimmed) {
      result.add(trimmed);
    }
  }
  return result;
}

function parseHostHeader(hostHeader) {
  if (!hostHeader) {
    return "";
  }

  try {
    return new URL(`http://${hostHeader}`).hostname.toLowerCase();
  } catch {
    return "";
  }
}

function isAllowedHost(hostname, allowedHosts) {
  return allowedHosts.has(hostname) || hostname.endsWith(".trycloudflare.com");
}

function getRequestId() {
  return randomUUID().slice(0, 8);
}

function timestamp() {
  return new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

function slugify(value, fallback = "automixer") {
  const slug = String(value || fallback)
    .toLowerCase()
    .replace(/[^a-z0-9_.-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
  return slug || fallback;
}

async function ensurePaperclipDirs() {
  await fs.mkdir(tasksDir, { recursive: true });
  await fs.mkdir(reportsDir, { recursive: true });
}

function assertAllowedWriteDir(dir) {
  const resolved = path.resolve(dir);
  if (!allowedWriteDirs.includes(resolved)) {
    throw new Error(`Refusing to write outside allowed artifact directories: ${dir}`);
  }
}

async function writeJsonFile(dir, prefix, label, payload) {
  assertAllowedWriteDir(dir);
  await ensurePaperclipDirs();
  const fileName = `${prefix}_${timestamp()}_${getRequestId()}_${slugify(label)}.json`;
  const filePath = path.join(dir, fileName);
  await fs.writeFile(filePath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  return filePath;
}

async function writeMarkdownReport(label, markdown) {
  assertAllowedWriteDir(reportsDir);
  await ensurePaperclipDirs();
  const fileName = `${timestamp()}_${getRequestId()}_${slugify(label)}_report.md`;
  const filePath = path.join(reportsDir, fileName);
  await fs.writeFile(filePath, markdown.endsWith("\n") ? markdown : `${markdown}\n`, "utf8");
  return filePath;
}

async function writeMarkdownReportWithPath(label, buildMarkdown) {
  assertAllowedWriteDir(reportsDir);
  await ensurePaperclipDirs();
  const fileName = `${timestamp()}_${getRequestId()}_${slugify(label)}_report.md`;
  const filePath = path.join(reportsDir, fileName);
  const reportFile = relativeToProject(filePath);
  const markdown = buildMarkdown(reportFile);
  const normalized = markdown.endsWith("\n") ? markdown : `${markdown}\n`;
  await fs.writeFile(filePath, normalized, "utf8");
  return { filePath, markdown: normalized };
}

function relativeToProject(filePath) {
  return path.relative(projectRoot, filePath);
}

function enforceDryRun(input) {
  return {
    requestedDryRun: input?.dryRun !== undefined ? Boolean(input.dryRun) : true,
    dryRun: true,
    dryRunEnforced: true,
    sourceCodeChangesAllowed: false
  };
}

async function runGit(args) {
  const key = args.join("\u0000");
  if (!allowedGitCommands.has(key)) {
    throw new Error(`Refusing non-allowlisted git command: git ${args.join(" ")}`);
  }

  const { stdout, stderr } = await execFileAsync("git", args, {
    cwd: projectRoot,
    encoding: "utf8",
    maxBuffer: 1024 * 1024
  });
  return {
    command: `git ${args.join(" ")}`,
    stdout: stdout.trim(),
    stderr: stderr.trim()
  };
}

async function collectGitSnapshot(includeLog = true) {
  const commands = [
    ["branch", "--show-current"],
    ["status", "--short", "--branch"],
    ["rev-parse", "--show-toplevel"]
  ];

  if (includeLog) {
    commands.push(["log", "-1", "--oneline"]);
  }

  const entries = [];
  for (const args of commands) {
    try {
      entries.push(await runGit(args));
    } catch (error) {
      entries.push({
        command: `git ${args.join(" ")}`,
        stdout: "",
        stderr: String(error?.stderr || error?.message || error)
      });
    }
  }
  return entries;
}

function gitSnapshotMarkdown(snapshot) {
  return snapshot
    .map((entry) => {
      const output = [entry.stdout, entry.stderr].filter(Boolean).join("\n");
      return `## ${entry.command}\n\n\`\`\`\n${output || "(no output)"}\n\`\`\``;
    })
    .join("\n\n");
}

function snapshotOutput(snapshot, command) {
  const entry = snapshot.find((item) => item.command === command);
  return [entry?.stdout, entry?.stderr].filter(Boolean).join("\n") || "(no output)";
}

function policyMarkdown() {
  return [
    "- dryRun: true",
    "- dryRunEnforced: true",
    "- sourceCodeChangesAllowed: false",
    `- writesAllowed: ${allowedWriteRelDirs.join(", ")}`,
    "- prohibitedActions: source code changes, dangerous shell commands, git commit, git checkout"
  ].join("\n");
}

function summarizeGitStatus(gitStatus) {
  const lines = String(gitStatus || "")
    .split(/\r?\n/)
    .map((line) => line.trimEnd())
    .filter(Boolean);
  const entries = lines.filter((line) => !line.startsWith("##"));
  let modified = 0;
  let untracked = 0;
  let deleted = 0;
  for (const line of entries) {
    const code = line.slice(0, 2);
    if (code.includes("M")) {
      modified += 1;
    }
    if (code.includes("D")) {
      deleted += 1;
    }
    if (line.startsWith("??")) {
      untracked += 1;
    }
  }

  if (entries.length === 0) {
    return "Git status is clean.";
  }

  return [
    `Git status has ${entries.length} changed or untracked entries.`,
    `${modified} modified, ${deleted} deleted, ${untracked} untracked.`
  ].join(" ");
}

function markdownTableCell(value) {
  return String(value)
    .replace(/\r?\n/g, "<br>")
    .replace(/\|/g, "\\|");
}

function modeDecision(mode) {
  if (mode === "implementation_plan") {
    return "Create an implementation plan only. Do not edit source code, run Codex as an executor, call external LLMs, commit, checkout, or apply changes.";
  }
  if (mode === "planning") {
    return "Create a planning report only. Keep the result as a Paperclip task/report artifact and return the report directly to GPT Chat.";
  }
  return "Create an analysis report only. Keep all work in dry-run mode and return the director report directly to GPT Chat.";
}

function buildDirectorRoleAssignments({ mode, prompt, gitBranch, gitStatusSummary }) {
  const promptLength = String(prompt || "").length;
  return [
    {
      role: "Director",
      assignment:
        "Coordinate the request, enforce dry-run policy, split work between internal roles, write the task/report artifacts, and return directorReport in the MCP response.",
      output: `${modeDecision(mode)} Branch: ${gitBranch || "(unknown)"}. Prompt length: ${promptLength} characters.`
    },
    {
      role: "DSP Agent",
      assignment:
        "Analyze possible DSP, routing, mixer, gain, EQ, dynamics, and live-console implications from the prompt and git status only.",
      output:
        "No DSP runtime action is allowed. Any future audio change must remain behind an explicit human approval gate and separate source-edit workflow."
    },
    {
      role: "ML Agent",
      assignment:
        "Analyze ML, evaluation, model, dataset, and scoring implications from the prompt and git status only.",
      output:
        "No model call, training run, dataset mutation, or external LLM call is part of this director execution."
    },
    {
      role: "Refactor Agent",
      assignment:
        "Identify implementation boundaries and refactor risks without modifying source code during director execution.",
      output:
        "The runtime director may create only Paperclip artifacts. Any future source refactor requires a separate approved task."
    },
    {
      role: "Analyzer Agent",
      assignment:
        "Summarize current git branch/status and capture repository context needed for a safe task plan.",
      output: gitStatusSummary
    },
    {
      role: "Safety Agent",
      assignment:
        "Validate dryRunEnforced=true, sourceCodeChangesAllowed=false, no Codex execution, no external LLM calls, no commit/checkout, and artifact-only writes.",
      output:
        "Policy enforced for this execution: dryRun=true, dryRunEnforced=true, sourceCodeChangesAllowed=false, codexAsExecutorAllowed=false, externalLlmAllowedForInternalRoles=false."
    }
  ];
}

function buildTaskBreakdown(mode) {
  return [
    `Classify request mode as ${mode}.`,
    "Capture the original GPT prompt exactly in the Paperclip task and report.",
    "Collect current git branch/status through the gateway allowlisted git snapshot helpers.",
    "Assign deterministic analysis work to Director, DSP Agent, ML Agent, Refactor Agent, Analyzer Agent, and Safety Agent.",
    "Write a task JSON under .paperclip/tasks.",
    "Write a Markdown director report under .paperclip/reports.",
    "Return both summary and directorReport directly in MCP structuredContent for GPT Chat."
  ];
}

function buildRiskAssessment(gitStatusSummary) {
  return [
    {
      risk: "Source code mutation",
      assessment:
        "Blocked by design. run_director does not call Codex, does not call external LLMs, and writes only Paperclip artifacts.",
      mitigation: "Keep sourceCodeChangesAllowed=false and require a separate explicit approval for code edits."
    },
    {
      risk: "Live audio or mixer side effects",
      assessment:
        "Not allowed. The director performs no mixer, OSC, WING, DSP, or automation writes.",
      mitigation: "Limit runtime actions to git snapshot reads and .paperclip artifact writes."
    },
    {
      risk: "Dirty working tree ambiguity",
      assessment: gitStatusSummary,
      mitigation: "Do not run git checkout/reset/commit. Treat existing changes as user-owned context."
    },
    {
      risk: "GPT Chat receives only file paths",
      assessment:
        "Addressed. MCP response includes both summary and the full directorReport string.",
      mitigation: "Keep directorReport in structuredContent and text content for the tool response."
    }
  ];
}

function buildExecutionPlan(mode) {
  if (mode === "implementation_plan") {
    return [
      "Keep this execution dry-run only.",
      "Use the role outputs to produce a source-edit plan, not a patch.",
      "Ask for explicit approval before any future code edit.",
      "After approval, restrict implementation to the approved files and run targeted checks."
    ];
  }

  if (mode === "planning") {
    return [
      "Keep this execution dry-run only.",
      "Convert the prompt into staged work items and acceptance criteria.",
      "Identify safety gates, tests, and artifact outputs required before implementation.",
      "Return the plan in directorReport for GPT Chat review."
    ];
  }

  return [
    "Keep this execution dry-run only.",
    "Analyze the prompt and repository state using internal deterministic role outputs.",
    "Record task/report artifacts for Paperclip auditability.",
    "Return the report in directorReport for GPT Chat."
  ];
}

function buildNextSteps(mode) {
  return [
    "Review the directorReport returned in GPT Chat.",
    "Open the created Paperclip task/report only if a persistent audit trail is needed.",
    "Do not treat this dry-run as approval for source changes.",
    mode === "implementation_plan"
      ? "For implementation, create a separate explicitly approved coding task with allowed file scope."
      : "For further work, call run_director again with planning or implementation_plan mode."
  ];
}

function buildDirectorReport({
  title,
  createdAt,
  prompt,
  mode,
  directorDecision,
  taskBreakdown,
  roleAssignments,
  riskAssessment,
  executionPlan,
  nextSteps,
  taskFile,
  reportFile,
  gitBranch,
  gitStatus,
  snapshot
}) {
  return [
    `# ${title}`,
    "",
    "## Title",
    "",
    title,
    "",
    "## Timestamp",
    "",
    createdAt,
    "",
    "## Original GPT Prompt",
    "",
    "```",
    prompt,
    "```",
    "",
    "## Director Decision",
    "",
    directorDecision,
    "",
    "## Task Breakdown",
    "",
    taskBreakdown.map((item, index) => `${index + 1}. ${item}`).join("\n"),
    "",
    "## Role Assignments",
    "",
    "| Role | Assignment | Structured Output |",
    "| --- | --- | --- |",
    ...roleAssignments.map(
      (item) =>
        `| ${markdownTableCell(item.role)} | ${markdownTableCell(item.assignment)} | ${markdownTableCell(item.output)} |`
    ),
    "",
    "## Risk Assessment",
    "",
    "| Risk | Assessment | Mitigation |",
    "| --- | --- | --- |",
    ...riskAssessment.map(
      (item) =>
        `| ${markdownTableCell(item.risk)} | ${markdownTableCell(item.assessment)} | ${markdownTableCell(item.mitigation)} |`
    ),
    "",
    "## Execution Plan",
    "",
    executionPlan.map((item, index) => `${index + 1}. ${item}`).join("\n"),
    "",
    "## Next Steps",
    "",
    nextSteps.map((item, index) => `${index + 1}. ${item}`).join("\n"),
    "",
    "## Created Task File Path",
    "",
    taskFile,
    "",
    "## Created Report File Path",
    "",
    reportFile,
    "",
    "## Dry Run Enforcement",
    "",
    "- dryRun: true",
    "- dryRunEnforced: true",
    "- sourceCodeChangesAllowed: false",
    "- codexAsExecutorAllowed: false",
    "- externalLlmAllowedForInternalRoles: false",
    `- mode: ${mode}`,
    "",
    "## Git Branch",
    "",
    "```",
    gitBranch,
    "```",
    "",
    "## Git Status",
    "",
    "```",
    gitStatus,
    "```",
    "",
    "## Full Git Snapshot",
    "",
    gitSnapshotMarkdown(snapshot)
  ].join("\n");
}

async function createTask(args, sourceTool) {
  const dryRun = enforceDryRun(args);
  const payload = {
    id: randomUUID(),
    source: "mcp_remote_gateway",
    sourceTool,
    projectRoot,
    createdAt: new Date().toISOString(),
    title: args.title || `${args.agent || "automixer"} task`,
    agent: args.agent || "automixer",
    prompt: args.prompt,
    metadata: args.metadata || {},
    ...dryRun,
    policy: gatewayPolicy
  };
  const taskFile = await writeJsonFile(tasksDir, "task", payload.agent, payload);
  return {
    ok: true,
    taskFile: relativeToProject(taskFile),
    taskId: payload.id,
    dryRun: payload.dryRun,
    dryRunEnforced: payload.dryRunEnforced,
    sourceCodeChangesAllowed: payload.sourceCodeChangesAllowed
  };
}

async function handleCreateTask(args) {
  const result = await createTask(args, "create_task");
  return {
    content: [
      {
        type: "text",
        text: `Created dry-run Paperclip task: ${result.taskFile}`
      }
    ],
    structuredContent: result
  };
}

async function handleRunAgent(args) {
  const dryRun = enforceDryRun(args);
  const task = await createTask(
    {
      ...args,
      title: `${args.agent || "automixer"} dry-run agent request`
    },
    "run_agent"
  );
  const snapshot = await collectGitSnapshot(true);
  const report = [
    `# ${(args.agent || "automixer").toUpperCase()} dry-run agent report`,
    "",
    "## Timestamp",
    "",
    new Date().toISOString(),
    "",
    "## Agent",
    "",
    args.agent || "automixer",
    "",
    "## Prompt",
    "",
    "```",
    args.prompt,
    "```",
    "",
    "## Dry Run Status",
    "",
    `dryRun: ${dryRun.dryRun}`,
    `dryRunEnforced: ${dryRun.dryRunEnforced}`,
    "",
    "## Policy",
    "",
    policyMarkdown(),
    "",
    "## Task File",
    "",
    task.taskFile,
    "",
    "## Git Branch",
    "",
    "```",
    snapshotOutput(snapshot, "git branch --show-current"),
    "```",
    "",
    "## Git Status",
    "",
    "```",
    snapshotOutput(snapshot, "git status --short --branch"),
    "```",
    "",
    "## Next Steps",
    "",
    "1. Open the task file in local Codex.",
    "2. Review the prompt and current git status manually.",
    "3. Run analysis only until a human explicitly approves code edits.",
    "4. Keep commits and branch changes outside this remote gateway.",
    "",
    "## Full Git Snapshot",
    "",
    gitSnapshotMarkdown(snapshot)
  ].join("\n");
  const reportFile = await writeMarkdownReport(args.agent || "automixer", report);
  const result = {
    ok: true,
    taskFile: task.taskFile,
    reportFile: relativeToProject(reportFile),
    dryRun: true,
    dryRunEnforced: true,
    sourceCodeChangesAllowed: false
  };
  return {
    content: [
      {
        type: "text",
        text: `Recorded dry-run agent request: ${result.taskFile}\nReport: ${result.reportFile}`
      }
    ],
    structuredContent: result
  };
}

async function handleRunDirector(args) {
  const dryRun = enforceDryRun(args);
  const createdAt = new Date().toISOString();
  const mode = args.mode || "analysis";
  const title = args.title || "Paperclip Director request";
  const snapshot = await collectGitSnapshot(true);
  const gitBranch = snapshotOutput(snapshot, "git branch --show-current");
  const gitStatus = snapshotOutput(snapshot, "git status --short --branch");
  const gitStatusSummary = summarizeGitStatus(gitStatus);
  const directorDecision = modeDecision(mode);
  const taskBreakdown = buildTaskBreakdown(mode);
  const roleAssignments = buildDirectorRoleAssignments({
    mode,
    prompt: args.prompt,
    gitBranch,
    gitStatusSummary
  });
  const riskAssessment = buildRiskAssessment(gitStatusSummary);
  const executionPlan = buildExecutionPlan(mode);
  const nextSteps = buildNextSteps(mode);
  const taskPayload = {
    id: randomUUID(),
    source: "mcp_remote_gateway",
    sourceTool: "run_director",
    projectRoot,
    createdAt,
    title,
    mode,
    prompt: args.prompt,
    requestedDryRun: dryRun.requestedDryRun,
    dryRun: true,
    dryRunEnforced: true,
    sourceCodeChangesAllowed: false,
    codexAsExecutorAllowed: false,
    externalLlmAllowedForInternalRoles: false,
    directorDecision,
    taskBreakdown,
    roleAssignments,
    riskAssessment,
    executionPlan,
    nextSteps,
    git: {
      branch: gitBranch,
      status: gitStatus,
      statusSummary: gitStatusSummary,
      snapshot
    },
    policy: gatewayPolicy
  };
  const taskFilePath = await writeJsonFile(tasksDir, "task", title, taskPayload);
  const taskFile = relativeToProject(taskFilePath);
  const { filePath: reportFilePath, markdown: directorReport } = await writeMarkdownReportWithPath(
    title,
    (reportFile) =>
      buildDirectorReport({
        title,
        createdAt,
        prompt: args.prompt,
        mode,
        directorDecision,
        taskBreakdown,
        roleAssignments,
        riskAssessment,
        executionPlan,
        nextSteps,
        taskFile,
        reportFile,
        gitBranch,
        gitStatus,
        snapshot
      })
  );
  const reportFile = relativeToProject(reportFilePath);
  const summary = [
    `Paperclip Director completed ${mode} dry-run.`,
    `Task: ${taskFile}.`,
    `Report: ${reportFile}.`,
    "No Codex executor, external LLM, source edit, commit, checkout, or live mixer action was used."
  ].join(" ");
  const result = {
    ok: true,
    taskFile,
    reportFile,
    summary,
    directorReport,
    dryRun: true,
    dryRunEnforced: true,
    sourceCodeChangesAllowed: false
  };
  return {
    content: [
      {
        type: "text",
        text: `${summary}\n\n${directorReport}`
      }
    ],
    structuredContent: result
  };
}

async function handleGitReport(args) {
  const snapshot = await collectGitSnapshot(args.includeLog !== false);
  const dryRun = enforceDryRun(args);
  const report = [
    `# ${args.title || "AUTO-MIXER git report"}`,
    "",
    `Created: ${new Date().toISOString()}`,
    "",
    `Dry run: ${dryRun.dryRun}`,
    "Source code changes allowed: false",
    "",
    gitSnapshotMarkdown(snapshot)
  ].join("\n");
  const reportFile = await writeMarkdownReport("git", report);
  const result = {
    ok: true,
    reportFile: relativeToProject(reportFile),
    dryRun: true,
    dryRunEnforced: true,
    sourceCodeChangesAllowed: false,
    snapshot
  };
  return {
    content: [
      {
        type: "text",
        text: `Created git report: ${result.reportFile}`
      }
    ],
    structuredContent: result
  };
}

async function listDirEntries(dir, kind, limit) {
  await ensurePaperclipDirs();
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    if (!entry.isFile()) {
      continue;
    }
    const filePath = path.join(dir, entry.name);
    const stats = await fs.stat(filePath);
    files.push({
      kind,
      file: relativeToProject(filePath),
      sizeBytes: stats.size,
      modifiedAt: stats.mtime.toISOString()
    });
  }
  return files
    .sort((a, b) => b.modifiedAt.localeCompare(a.modifiedAt))
    .slice(0, limit);
}

async function handleListArtifacts(args) {
  const limit = args.limit || 25;
  let tasks = [];
  let reports = [];
  if (args.kind === "all" || args.kind === "tasks") {
    tasks = await listDirEntries(tasksDir, "tasks", limit);
  }
  if (args.kind === "all" || args.kind === "reports") {
    reports = await listDirEntries(reportsDir, "reports", limit);
  }
  const artifacts = [...tasks, ...reports]
    .sort((a, b) => b.modifiedAt.localeCompare(a.modifiedAt))
    .slice(0, limit);
  const result = {
    ok: true,
    artifacts,
    tasks,
    reports,
    count: artifacts.length,
    dryRun: true,
    dryRunEnforced: true,
    sourceCodeChangesAllowed: false
  };
  return {
    content: [
      {
        type: "text",
        text: artifacts.length
          ? artifacts.map((artifact) => artifact.file).join("\n")
          : "No Paperclip artifacts found."
      }
    ],
    structuredContent: result
  };
}

function createMcpServer() {
  const server = new McpServer(
    {
      name: "automixer-paperclip-gateway",
      version: "1.0.0"
    },
    {
      capabilities: {
        logging: {}
      }
    }
  );

  server.registerTool(
    "create_task",
    {
      title: "Create Paperclip Task",
      description:
        "Use this when ChatGPT needs to record a supervised AUTO-MIXER task. Writes only to .paperclip/tasks and always enforces dry-run behavior.",
      inputSchema: createTaskInput,
      outputSchema: taskOutput,
      annotations: {
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: false,
        readOnlyHint: false
      },
      _meta: {
        "openai/toolInvocation/invoking": "Creating dry-run task.",
        "openai/toolInvocation/invoked": "Dry-run task created."
      }
    },
    handleCreateTask
  );

  server.registerTool(
    "run_agent",
    {
      title: "Record Dry-Run Agent Request",
      description:
        "Use this when ChatGPT needs to queue an AUTO-MIXER agent request without changing source code. Creates a task and a dry-run report only.",
      inputSchema: runAgentInput,
      outputSchema: runAgentOutput,
      annotations: {
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: false,
        readOnlyHint: false
      },
      _meta: {
        "openai/toolInvocation/invoking": "Recording dry-run agent request.",
        "openai/toolInvocation/invoked": "Dry-run agent request recorded."
      }
    },
    handleRunAgent
  );

  server.registerTool(
    "run_director",
    {
      title: "Run Paperclip Director",
      description:
        "Use this when ChatGPT needs the Paperclip Director to distribute a dry-run task to internal deterministic roles, create task/report artifacts, and return the director report directly in the MCP response. Does not run Codex or external LLMs.",
      inputSchema: runDirectorInput,
      outputSchema: runDirectorOutput,
      annotations: {
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: false,
        readOnlyHint: false
      },
      _meta: {
        "openai/toolInvocation/invoking": "Running Paperclip Director dry-run.",
        "openai/toolInvocation/invoked": "Paperclip Director report returned."
      }
    },
    handleRunDirector
  );

  server.registerTool(
    "git_report",
    {
      title: "Create Git Report",
      description:
        "Use this when ChatGPT needs a read-only git status snapshot. Writes the snapshot as a markdown report in .paperclip/reports.",
      inputSchema: gitReportInput,
      outputSchema: gitReportOutput,
      annotations: {
        destructiveHint: false,
        idempotentHint: false,
        openWorldHint: false,
        readOnlyHint: false
      },
      _meta: {
        "openai/toolInvocation/invoking": "Creating git report.",
        "openai/toolInvocation/invoked": "Git report created."
      }
    },
    handleGitReport
  );

  server.registerTool(
    "list_artifacts",
    {
      title: "List Paperclip Artifacts",
      description:
        "Use this when ChatGPT needs to inspect recent Paperclip tasks and reports created by the gateway.",
      inputSchema: listArtifactsInput,
      outputSchema: listArtifactsOutput,
      annotations: {
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
        readOnlyHint: true
      },
      _meta: {
        "openai/toolInvocation/invoking": "Listing Paperclip artifacts.",
        "openai/toolInvocation/invoked": "Paperclip artifacts listed."
      }
    },
    handleListArtifacts
  );

  return server;
}

export function createAutomixerMcpApp(options = {}) {
  const host = options.host || process.env.HOST || "127.0.0.1";
  const sessions = new Map();
  const allowedHosts = parseCsvList(process.env.ALLOWED_HOSTS, defaultAllowedHosts);
  const app = createMcpExpressApp({ host: "mcp-gateway.local" });

  app.set("trust proxy", options.trustProxy ?? true);
  app.use((req, res, next) => {
    const requestHost = parseHostHeader(req.get("host"));
    if (!requestHost || !isAllowedHost(requestHost, allowedHosts)) {
      res.status(403).json({
        jsonrpc: "2.0",
        error: {
          code: -32000,
          message: `Invalid Host: ${requestHost || req.get("host") || "missing"}`
        },
        id: null
      });
      return;
    }

    const origin = req.get("origin");
    res.setHeader("Access-Control-Allow-Origin", origin || "*");
    res.setHeader("Vary", "Origin");
    res.setHeader(
      "Access-Control-Allow-Headers",
      "Content-Type, Accept, Mcp-Session-Id, MCP-Protocol-Version, Last-Event-ID"
    );
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    res.setHeader("Access-Control-Expose-Headers", "Mcp-Session-Id, MCP-Protocol-Version");
    res.setHeader("X-Content-Type-Options", "nosniff");
    next();
  });

  app.options(MCP_PATH, (_req, res) => {
    res.sendStatus(204);
  });

  app.get("/health", (_req, res) => {
    res.json({
      ok: true,
      name: "automixer-paperclip-gateway",
      mcpPath: MCP_PATH,
      sessions: sessions.size,
      dryRun: true,
      dryRunEnforced: true,
      sourceCodeChangesAllowed: false
    });
  });

  const createSession = async () => {
    const server = createMcpServer();
    const eventStore = new InMemoryEventStore();
    let transport;

    transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: () => randomUUID(),
      eventStore,
      onsessioninitialized: (sessionId) => {
        sessions.set(sessionId, {
          server,
          transport,
          createdAt: new Date().toISOString()
        });
      }
    });

    transport.onerror = (error) => {
      console.error("MCP transport error:", error);
    };

    transport.onclose = () => {
      const sessionId = transport.sessionId;
      if (sessionId) {
        sessions.delete(sessionId);
      }
    };

    await server.connect(transport);
    return { server, transport, createdAt: new Date().toISOString() };
  };

  app.post(MCP_PATH, async (req, res) => {
    try {
      const sessionId = req.get("mcp-session-id");
      let session = sessionId ? sessions.get(sessionId) : undefined;

      if (sessionId && !session) {
        res.status(404).json({
          jsonrpc: "2.0",
          error: { code: -32001, message: "Session not found" },
          id: null
        });
        return;
      }

      if (!session) {
        if (!isInitializeRequest(req.body)) {
          res.status(400).json({
            jsonrpc: "2.0",
            error: { code: -32000, message: "Bad Request: No valid session ID provided" },
            id: null
          });
          return;
        }

        session = await createSession();
      }

      await session.transport.handleRequest(req, res, req.body);
    } catch (error) {
      console.error("MCP POST error:", error);
      if (!res.headersSent) {
        res.status(500).json({
          jsonrpc: "2.0",
          error: { code: -32603, message: "Internal server error" },
          id: null
        });
      }
    }
  });

  const handleSessionRequest = async (req, res) => {
    try {
      const sessionId = req.get("mcp-session-id");
      const session = sessionId ? sessions.get(sessionId) : undefined;
      if (!session) {
        res.status(400).json({
          jsonrpc: "2.0",
          error: { code: -32000, message: "Invalid or missing session ID" },
          id: null
        });
        return;
      }

      await session.transport.handleRequest(req, res);
    } catch (error) {
      console.error(`MCP ${req.method} error:`, error);
      if (!res.headersSent) {
        res.status(500).json({
          jsonrpc: "2.0",
          error: { code: -32603, message: "Internal server error" },
          id: null
        });
      }
    }
  };

  app.get(MCP_PATH, handleSessionRequest);
  app.delete(MCP_PATH, handleSessionRequest);

  app.use((_req, res) => {
    res.status(404).json({
      jsonrpc: "2.0",
      ok: false,
      error: {
        code: -32000,
        message: "Not found"
      },
      id: null,
      mcpPath: MCP_PATH
    });
  });

  return app;
}

export async function startServer(options = {}) {
  await ensurePaperclipDirs();
  const host = options.host || process.env.HOST || "127.0.0.1";
  const port = Number(options.port || process.env.PORT || PORT);
  const app = createAutomixerMcpApp(options);
  return app.listen(port, host, () => {
    console.log(`MCP URL: http://${host}:${port}${MCP_PATH}`);
    console.log(`Public ChatGPT URL: ${PUBLIC_MCP_URL}`);
    console.log("Dry run enforced: true");
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  startServer().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
