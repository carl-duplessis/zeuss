# Coding Zeuss from your phone (Claude Code Remote Control)

Goal: run Claude Code **on this Windows machine** (where the code and your GPU
live) and steer it from the **Claude app on your phone** — start a task at your
desk, continue it from the couch. The code never leaves your machine; the phone
is just a window into the local session.

> Requires a Claude **Pro or Max** subscription (also Team/Enterprise, where an
> Owner must enable Remote Control first). API-key logins are not supported.

## 1. Install Claude Code (one time)

You already have Node 22 installed. In a terminal:

```powershell
npm install -g @anthropic-ai/claude-code
claude --version
```

If native Windows gives you trouble, run Claude Code inside **WSL2** instead
(also the smoother path for later JAX/Triton GPU work).

## 2. Sign in with your Claude account (one time)

```powershell
claude          # starts Claude Code
/login          # choose "Claude account" and finish in the browser
/status         # confirm the Login method row shows your claude.ai account
```

Remote Control needs the claude.ai login, not an API key.

## 3. Trust the project folder (one time)

Start Claude Code once **from this project folder** so it records workspace
trust (the trust prompt is never saved for your home directory, so start it
from here):

```powershell
cd C:\Program
claude           # accept the workspace-trust prompt, then /exit
```

## 4. Start a Remote Control session

From the project folder, pick one:

```powershell
cd C:\Program

# Server mode — stays running, waits for your phone/browser to connect:
claude remote-control --name "Zeuss"
# press SPACE to show a QR code for your phone

# …or an interactive session you can also type in locally:
claude --remote-control "Zeuss"
```

You can also show the pairing QR at any time from inside Claude Code with:

```
/mobile
```

## 5. Connect from your phone

Install the Claude app and open the **Code** tab:

- iOS: https://apps.apple.com/us/app/claude-by-anthropic/id6473753684
- Android: https://play.google.com/store/apps/details?id=com.anthropic.claude

Scan the QR code (or open the session from the list at **claude.ai/code**). Your
phone, your terminal, and any browser stay in sync — send messages from any of
them. If your laptop sleeps or the network drops, Claude Code reconnects and
delivers any queued messages when the machine is back.

Type `@` on the phone to autocomplete real file paths from this project, and
attach photos/files from your phone straight into the session.

---

## Alternative: Claude Code on the web (cloud, no local machine)

If you'd rather have sessions run in the cloud (e.g. your PC is off), use
**Claude Code on the web**. It runs in an Anthropic-managed VM tied to a GitHub
repo and is reachable from the same **claude.ai/code** / Code tab.

Trade-off: it needs this repo pushed to **GitHub**, and it won't have your local
GPU or local config — it works from the repo only. Set-up:

1. Push Zeuss to a GitHub repo (see below).
2. Go to **claude.ai/code**, connect GitHub, let it create the **Default**
   environment.
3. Pick the repo, describe a task; Claude works in a branch and opens a PR.

Or, from this machine's terminal (you have the `gh` CLI flow available):

```powershell
gh auth login
claude
/web-setup
```

### Push Zeuss to GitHub (needed only for the web option)

```powershell
cd C:\Program
gh repo create zeuss --private --source . --push
# or manually:
#   git remote add origin https://github.com/<you>/zeuss.git
#   git push -u origin main
```

**Which should you use?** Remote Control (local) matches "run it in Claude Code
and code from my phone" best and keeps everything on your machine. Use Claude
Code on the web when you want cloud sessions you can fire off and review later
without your PC being on.

Docs: https://code.claude.com/docs/en/remote-control ·
https://code.claude.com/docs/en/web-quickstart
