#!/usr/bin/env node
import fs from "fs";
import path from "path";
import { execSync } from "child_process";
import { z } from "zod";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const PROJECT_DIR = process.env.AUTOMIXER_PROJECT_DIR || process.cwd();
const PAPERCLIP_DIR = path.join(PROJECT_DIR, ".paperclip");
const TASKS_DIR = path.join(PAPERCLIP_DIR, "tasks");
const REPORTS_DIR = path.join(PAPERCLIP_DIR, "reports");

fs.mkdirSync(TASKS_DIR, { recursive: true });
fs.mkdirSync(REPORTS_DIR, { recursive: true });

function nowId() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function writeJson(file, data) {
  fs.writeFileSync(file, JSON.stringify(data, null, 2), "utf8");
}

function writeMd(file, text) {
  fs.writeFileSync(file, text, "utf8");
}

function sh(cmd) {
  return execSync(cmd, {
    cwd: PROJECT_DIR,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"]
  });
}

function safeBranchName(title) {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9а-яё]+/gi, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 60);
}

const AGENTS = {
  dsp: "DSP Agent: analyze signal-processing, EQ, dynamics, latency, OSC safety, live constraints.",
  ml: "ML Agent: analyze ML/AI critic models, datasets, offline-to-live transfer, evaluation strategy.",
  refactor: "Refactor Agent: find dead code, unsafe coupling, duplicated logic, architecture cleanup.",
  analyzer: "Analyzer Agent: inspect analyzers, metrics, logs, spectral balance and decision quality."
};

const server = new McpServer({
  name: "automixer-paperclip-director",
  version: "0.1.0"
});

server.tool(
  "create_task",
  {
    title: z.string(),
    description: z.string(),
    priority: z.enum(["low", "normal", "high"]).default("normal"),
    agent: z.enum(["dsp", "ml", "refactor", "analyzer", "director"]).default("director")
  },
  async ({ title, description, priority, agent }) => {
    const id = nowId();
    const task = {
      id,
      title,
      description,
      priority,
      agent,
      status: "created",
      created_at: new Date().toISOString()
    };

    const file = path.join(TASKS_DIR, `${id}-${agent}.json`);
    writeJson(file, task);

    return {
      content: [
        {
          type: "text",
          text: `Task created: ${file}`
        }
      ]
    };
  }
);

server.tool(
  "run_agent",
  {
    agent: z.enum(["dsp", "ml", "refactor", "analyzer"]),
    task: z.string(),
    dry_run: z.boolean().default(true)
  },
  async ({ agent, task, dry_run }) => {
    const id = nowId();
    const branch = `agent/${agent}-${safeBranchName(task)}`;
    const reportFile = path.join(REPORTS_DIR, `${id}-${agent}-report.md`);

    let gitInfo = "";
    try {
      gitInfo = sh("git status --short");
    } catch {
      gitInfo = "Git status unavailable.";
    }

    const prompt = `
# ${AGENTS[agent]}

Project: AUTO-MIXER-Tubeslave

Task:
${task}

Mode:
${dry_run ? "DRY RUN. Do not modify files. Produce analysis and proposed patch plan only." : "Implementation allowed, but only on a separate git branch."}

Required output:
1. Findings
2. Risks
3. Concrete implementation plan
4. Files likely affected
5. Tests to run
6. Rollback plan

Current git status:
${gitInfo}
`;

    writeMd(reportFile, prompt);

    if (!dry_run) {
      try {
        sh(`git checkout -b ${branch}`);
      } catch {
        // Branch may already exist.
      }
    }

    return {
      content: [
        {
          type: "text",
          text: `Agent task prepared.\nAgent: ${agent}\nBranch: ${dry_run ? "not created, dry-run" : branch}\nReport: ${reportFile}\n\nOpen this report in Codex and let the agent execute it.`
        }
      ]
    };
  }
);

server.tool(
  "run_full_review",
  {
    topic: z.string(),
    dry_run: z.boolean().default(true)
  },
  async ({ topic, dry_run }) => {
    const id = nowId();
    const reportFile = path.join(REPORTS_DIR, `${id}-full-review.md`);

    const sections = Object.entries(AGENTS)
      .map(([name, role]) => `## ${name.toUpperCase()}\n${role}\n\nReview topic: ${topic}\n`)
      .join("\n");

    const text = `# Automixer Multi-Agent Review

Topic:
${topic}

Mode:
${dry_run ? "DRY RUN. No code changes." : "Implementation may be proposed, but each change must be isolated in branches."}

${sections}

# Final Director Decision

After all agents respond, synthesize:
1. Best idea
2. What not to implement
3. Safe first patch
4. Tests
5. Risk level for live concert use
`;

    writeMd(reportFile, text);

    return {
      content: [
        {
          type: "text",
          text: `Full multi-agent review prepared: ${reportFile}`
        }
      ]
    };
  }
);

server.tool(
  "git_report",
  {
    include_diff: z.boolean().default(false)
  },
  async ({ include_diff }) => {
    const id = nowId();
    const reportFile = path.join(REPORTS_DIR, `${id}-git-report.md`);

    let status = "";
    let branch = "";
    let log = "";
    let diff = "";

    try { status = sh("git status --short"); } catch { status = "Unavailable"; }
    try { branch = sh("git branch --show-current").trim(); } catch { branch = "Unavailable"; }
    try { log = sh("git log --oneline -10"); } catch { log = "Unavailable"; }
    if (include_diff) {
      try { diff = sh("git diff --stat"); } catch { diff = "Unavailable"; }
    }

    const text = `# Git Report

Branch:
${branch}

Status:
\`\`\`
${status}
\`\`\`

Last commits:
\`\`\`
${log}
\`\`\`

${include_diff ? `Diff stat:\n\`\`\`\n${diff}\n\`\`\`` : ""}
`;

    writeMd(reportFile, text);

    return {
      content: [
        {
          type: "text",
          text: `Git report created: ${reportFile}`
        }
      ]
    };
  }
);

server.tool(
  "telegram_notify",
  {
    message: z.string()
  },
  async ({ message }) => {
    const token = process.env.TELEGRAM_BOT_TOKEN;
    const chatId = process.env.TELEGRAM_CHAT_ID;

    if (!token || !chatId) {
      return {
        content: [
          {
            type: "text",
            text: "Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
          }
        ]
      };
    }

    const url = `https://api.telegram.org/bot${token}/sendMessage`;
    const payload = JSON.stringify({
      chat_id: chatId,
      text: message
    });

    const cmd = `curl -s -X POST '${url}' -H 'Content-Type: application/json' -d '${payload.replace(/'/g, "'\\''")}'`;
    const result = sh(cmd);

    return {
      content: [
        {
          type: "text",
          text: `Telegram notification sent.\n${result}`
        }
      ]
    };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
