#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, sys, datetime

STATE_FILE = ".ai/state.json"
TASKS_FILE = ".ai/tasks.json"
LOG_FILE   = "ai/session_log.md"

ROLES = ["ARCHITECT","PLANNER","CODE","REVIEWER","TESTER","CRITIC"]

def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        try: return json.load(f)
        except: return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def log(msg):
    os.makedirs("ai", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{ts}] {msg}\n")
    print(msg)

def print_prompt(role, task, extra=""):
    base = f"""
ROLE: {role}

Project context:
- Read .ai/system_prompt.md, .ai/personas.md, .ai/rules.md
- Read project_docs/*

Task:
{task}

Instructions:
- Follow your role strictly
- Be concise but precise
- Do NOT implement code unless role == CODE and state.approved == true
"""
    if role == "ARCHITECT":
        base += """
Output:
1) Problem understanding
2) 2–3 solution approaches
3) Trade-offs
4) Recommendation
5) WAIT for approval
"""
    elif role == "PLANNER":
        base += """
Output:
- Step-by-step plan
- JSON tasks list
"""
    elif role == "CODE":
        base += """
Output:
- Exact files to change
- Patches/snippets
"""
    elif role == "REVIEWER":
        base += """
Output:
- Risks
- Bugs
- Architecture issues
"""
    elif role == "TESTER":
        base += """
Output:
- How to run tests
- Expected results
"""
    elif role == "CRITIC":
        base += """
Output:
- Audio/mix critique (balance, masking, dynamics)
- What improved / what broke
"""
    if extra:
        base += f"\nExtra:\n{extra}\n"
    print("="*80)
    print(base.strip())
    print("="*80)

def ensure_files():
    os.makedirs(".ai", exist_ok=True)
    if not os.path.exists(STATE_FILE):
        save_json(STATE_FILE, {"status":"idle","last_step":None,"task":None,"approved":False})
    if not os.path.exists(TASKS_FILE):
        save_json(TASKS_FILE, [])

def start(task):
    state = {"status":"running","last_step":"ARCHITECT","task":task,"approved":False}
    save_json(STATE_FILE, state)
    log(f"START task: {task}")
    print_prompt("ARCHITECT", task)

def approve():
    state = load_json(STATE_FILE, {})
    if state.get("status") != "running":
        print("No active task."); return
    state["approved"] = True
    state["last_step"] = "CODE"
    save_json(STATE_FILE, state)
    log("APPROVED by user → moving to CODE")
    print_prompt("CODE", state["task"], extra="Implement according to approved approach.")

def next_step():
    state = load_json(STATE_FILE, {})
    if state.get("status") != "running":
        print("No active task."); return

    step = state.get("last_step")
    task = state.get("task")

    order = ["ARCHITECT","PLANNER","CODE","REVIEWER","TESTER","CRITIC"]
    if step not in order:
        print("Invalid state."); return

    idx = order.index(step)

    # After ARCHITECT → go to PLANNER
    if step == "ARCHITECT":
        state["last_step"] = "PLANNER"
        save_json(STATE_FILE, state)
        log("NEXT → PLANNER")
        print_prompt("PLANNER", task)
        return

    # After PLANNER → wait for approval before CODE
    if step == "PLANNER":
        if not state.get("approved"):
            log("WAITING FOR APPROVAL. Run: python ai/orchestrator.py approve")
            print_prompt("ARCHITECT", task, extra="User must approve before implementation.")
            return
        else:
            state["last_step"] = "CODE"
            save_json(STATE_FILE, state)
            log("NEXT → CODE")
            print_prompt("CODE", task)
            return

    # After CODE → REVIEWER → TESTER → CRITIC → loop to ARCHITECT (iteration)
    if step == "CODE":
        state["last_step"] = "REVIEWER"
    elif step == "REVIEWER":
        state["last_step"] = "TESTER"
    elif step == "TESTER":
        state["last_step"] = "CRITIC"
    elif step == "CRITIC":
        state["last_step"] = "ARCHITECT"
        state["approved"] = False  # require re-approval for next iteration

    save_json(STATE_FILE, state)
    log(f"NEXT → {state['last_step']}")
    print_prompt(state["last_step"], task)

def status():
    state = load_json(STATE_FILE, {})
    print(json.dumps(state, ensure_ascii=False, indent=2))

def reset():
    save_json(STATE_FILE, {"status":"idle","last_step":None,"task":None,"approved":False})
    log("RESET")
    print("State reset.")

def help_msg():
    print("""
Usage:
  python ai/orchestrator.py start "task text"
  python ai/orchestrator.py next
  python ai/orchestrator.py approve
  python ai/orchestrator.py status
  python ai/orchestrator.py reset
""")

if __name__ == "__main__":
    ensure_files()
    if len(sys.argv) < 2:
        help_msg(); sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "start":
        task = " ".join(sys.argv[2:]).strip()
        if not task:
            print("Provide task."); sys.exit(1)
        start(task)
    elif cmd == "next":
        next_step()
    elif cmd == "approve":
        approve()
    elif cmd == "status":
        status()
    elif cmd == "reset":
        reset()
    else:
        help_msg()
