// Electron 主进程（生产打包入口）：以本地 HTTP 服务方式启动 PureCut Python 引擎
// （scripts/web_studio.py），然后在无边框窗口中加载其 UI。
// 关键点：
//  - Python 引擎通过 spawn 启动，端口动态获取（不写死），轮询就绪后再 loadURL
//  - ffmpeg/ffprobe 使用打包内置的二进制，经 FFMPEG_PATH/FFPROBE_PATH 注入（用户机器无需安装 ffmpeg）
//  - 打包 GUI 无控制台，子进程 stdout/stderr 全部汇入 userData/logs/server.log，启动失败弹窗可见
//  - 无边框窗口 + preload 桥接（窗口控制 / 原生目录选择对话框）
const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const net = require("net");
const path = require("path");
const fs = require("fs");

let serverChild = null;
let mainWindow = null;
let childExited = false; // 子进程一旦报错/退出就置位，waitReady 立即失败而不是死等
let childExitInfo = "";
let logFilePath = "";

/** 初始化诊断日志（每次启动截断一次）。打包 GUI 无控制台，一切启动错误都汇入这里。 */
function initLog() {
  try {
    const logDir = path.join(app.getPath("userData"), "logs");
    fs.mkdirSync(logDir, { recursive: true });
    logFilePath = path.join(logDir, "server.log");
    fs.writeFileSync(logFilePath, `=== PureCut 启动 ${new Date().toISOString()} (${process.platform}/${process.arch}) ===\n`);
  } catch {
    logFilePath = "";
  }
}

function log(line) {
  if (!logFilePath) return;
  try {
    fs.appendFileSync(logFilePath, `[main] ${line}\n`);
  } catch {
    /* ignore */
  }
}

function readLogTail(maxChars = 3000) {
  if (!logFilePath) return "";
  try {
    return fs.readFileSync(logFilePath, "utf8").slice(-maxChars);
  } catch {
    return "";
  }
}

/** 获取一个空闲的本地端口 */
function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

/** 修正 asar → asar.unpacked，取到内置二进制（ffmpeg/ffprobe）的真实路径 */
function resolveBinary(getter) {
  try {
    let p = getter();
    if (p && p.includes("app.asar" + path.sep)) {
      p = p.replace("app.asar" + path.sep, "app.asar.unpacked" + path.sep);
    }
    return p || "";
  } catch {
    return "";
  }
}

/**
 * 解析引擎资源根目录（含 scripts/ 与 models/）。支持热更新：外部可写目录优先于打包内置。
 * web_studio.py 以 __file__.parent.parent 作为 REPO_ROOT，从根目录定位 models/ 与 temp_web_studio/，
 * 所以只要把「引擎根目录」整体指向某个包含 scripts/ + models/ 的目录即可正确加载。
 */
function engineRoot() {
  // 1) 显式环境变量（可指向任意开发/更新目录）
  const env = process.env.PURECUT_ENGINE_DIR;
  if (env && fs.existsSync(path.join(env, "scripts", "web_studio.py"))) return env;
  // 2) 热更新目录 userData/engine（可写；替换其中的 scripts/ + models/ 后重启即生效，无需重新打包）
  const hot = path.join(app.getPath("userData"), "engine");
  if (fs.existsSync(path.join(hot, "scripts", "web_studio.py"))) return hot;
  // 3) 打包内置 resources / 开发项目根
  return app.isPackaged ? process.resourcesPath : path.join(__dirname, "..");
}

function pythonEntry() {
  return path.join(engineRoot(), "scripts", "web_studio.py");
}

function pythonCwd() {
  return engineRoot();
}

/**
 * 轮询 HTTP 直到引擎就绪。Python 引擎冷启动（含 MediaPipe / ONNX 模型加载）较慢，
 * 超时预算给到 30s（tries=120 × 250ms）。子进程若先退出则立即失败，避免空转死等。
 */
function waitReady(port, tries = 120) {
  return new Promise((resolve, reject) => {
    const attempt = (n) => {
      if (childExited) return reject(new Error("本地引擎进程已退出：" + childExitInfo));
      const req = http.get({ host: "127.0.0.1", port, path: "/api/settings/directories" }, (res) => {
        res.resume();
        if (res.statusCode && res.statusCode < 500) return resolve();
        retry(n);
      });
      req.on("error", () => retry(n));
    };
    const retry = (n) => (n <= 0 ? reject(new Error("本地引擎未就绪（超时）")) : setTimeout(() => attempt(n - 1), 250));
    attempt(tries);
  });
}

/** 启动 Python 引擎子进程，等待就绪，返回访问 URL */
async function startServer() {
  const entry = pythonEntry();
  const cwd = pythonCwd();
  const port = await getFreePort();

  const ffmpegPath = resolveBinary(() => require("ffmpeg-static"));
  const ffprobePath = resolveBinary(() => require("@ffprobe-installer/ffprobe").path);

  // 子进程 stdout/stderr 通过真实文件描述符汇入日志。关键：Windows 打包 GUI 无控制台，
  // 继承的 stdout/stderr 句柄无效，Python 打印 banner 时 EBADF/EPIPE 会让引擎在绑端口前崩溃，
  // 用户感知就是「点了没反应」。文件描述符永远是合法写入目标。
  let outFd = "ignore";
  if (logFilePath) {
    try {
      outFd = fs.openSync(logFilePath, "a");
    } catch {
      outFd = "ignore";
    }
  }

  serverChild = spawn("python", [entry, String(port)], {
    cwd,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      ...(ffmpegPath ? { FFMPEG_PATH: ffmpegPath } : {}),
      ...(ffprobePath ? { FFPROBE_PATH: ffprobePath } : {}),
    },
    stdio: ["ignore", outFd, outFd],
  });

  serverChild.on("error", (err) => {
    childExited = true;
    childExitInfo = "spawn error: " + err.message;
    log(childExitInfo);
  });
  serverChild.on("exit", (code, signal) => {
    childExited = true;
    childExitInfo = `引擎进程退出 code=${code} signal=${signal}`;
    log(childExitInfo);
  });

  log(`spawn python pid=${serverChild.pid} port=${port} entry=${entry}`);
  await waitReady(port);
  return `http://127.0.0.1:${port}`;
}

function killServer() {
  if (serverChild) {
    try {
      serverChild.kill();
    } catch {
      /* ignore */
    }
    serverChild = null;
  }
}

/** 重启引擎（热更新 .py 时调用）：杀旧进程 → 重新 spawn → 刷新窗口 */
async function restartServer() {
  killServer();
  childExited = false;
  childExitInfo = "";
  const url = await startServer();
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.loadURL(url);
}

/**
 * 开发模式热重载：监听引擎 scripts/ 目录。
 *  - .html 改动 → 仅刷新页面（web_studio.py 每次请求都重新读 studio_template.html）
 *  - .py 改动 → 重启 Python 引擎后刷新
 */
function watchEngine() {
  const scriptsDir = path.join(engineRoot(), "scripts");
  let timer = null;
  try {
    fs.watch(scriptsDir, { recursive: true }, (_event, filename) => {
      if (!filename) return;
      clearTimeout(timer);
      timer = setTimeout(() => {
        const name = String(filename);
        log(`[hot-reload] 检测到改动: ${name}`);
        if (name.endsWith(".html")) {
          if (mainWindow && !mainWindow.isDestroyed()) mainWindow.reload();
        } else if (name.endsWith(".py")) {
          restartServer().catch((e) => log("热重载重启失败: " + (e && e.message)));
        }
      }, 400);
    });
    log("[hot-reload] 已启用（开发模式），改动 scripts/ 会自动刷新 / 重启引擎");
  } catch (e) {
    log("[hot-reload] 启用失败: " + (e && e.message));
  }
}

// ---- 窗口控制与原生对话框 IPC（与 preload.js 暴露的 electronAPI 一一对应）----
ipcMain.on("window-minimize", () => {
  if (mainWindow) mainWindow.minimize();
});
ipcMain.on("window-maximize", () => {
  if (mainWindow) {
    if (mainWindow.isMaximized()) mainWindow.unmaximize();
    else mainWindow.maximize();
  }
});
ipcMain.handle("window-is-maximized", () => (mainWindow ? mainWindow.isMaximized() : false));
ipcMain.on("window-close", () => {
  if (mainWindow) mainWindow.close();
});
ipcMain.handle("dialog-select-directory", async (_event, defaultPath) => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "选择包含视频素材的文件夹",
    defaultPath: defaultPath || undefined,
    properties: ["openDirectory", "dontAddToRecent"],
  });
  return result.canceled || result.filePaths.length === 0 ? null : result.filePaths[0];
});
ipcMain.handle("dialog-select-videos", async () => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "选择视频素材文件",
    filters: [
      { name: "视频文件", extensions: ["mp4", "mov", "webm", "mkv", "avi", "m4v", "flv", "wmv"] },
      { name: "所有文件", extensions: ["*"] },
    ],
    properties: ["openFile", "multiSelections", "dontAddToRecent"],
  });
  return result.canceled || result.filePaths.length === 0 ? null : result.filePaths;
});
ipcMain.on("shell-show-item", (_event, fullPath) => {
  if (fullPath) shell.showItemInFolder(fullPath);
});
ipcMain.handle("shell-open-path", async (_event, fullPath) => {
  if (fullPath) return await shell.openPath(fullPath);
  return "";
});

app.whenReady().then(async () => {
  initLog();
  let url;
  try {
    url = await startServer();
  } catch (e) {
    const msg = (e && (e.stack || e.message)) || String(e);
    log("启动本地引擎失败: " + msg);
    dialog.showErrorBox(
      "PureCut 启动失败 / Failed to start",
      `本地引擎未能启动 / The local engine failed to start.\n\n${(e && e.message) || e}\n\n` +
        `日志 / Log: ${logFilePath || "(不可用)"}\n\n--- 日志末尾 / log tail ---\n${readLogTail()}`
    );
    app.quit();
    return;
  }

  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1100,
    minHeight: 720,
    frame: false, // 无边框，配合 preload 的自定义标题栏控制
    backgroundColor: "#0c0d10",
    title: "PureCut",
    icon: path.join(__dirname, "icon.png"),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false, // 允许本地文件流式访问
    },
  });

  mainWindow.webContents.on("did-fail-load", (_e, errorCode, errorDescription, validatedURL) => {
    if (errorCode === -3) return; // ERR_ABORTED：良性（重定向/导航），忽略
    log(`页面加载失败 code=${errorCode} desc=${errorDescription} url=${validatedURL}`);
    dialog.showErrorBox(
      "PureCut 页面加载失败 / Page load failed",
      `${errorDescription} (${errorCode})\n${validatedURL}\n\n日志 / Log: ${logFilePath || "(不可用)"}`
    );
  });

  mainWindow.loadURL(url);
  if (!app.isPackaged) watchEngine();
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
});

app.on("window-all-closed", () => {
  killServer();
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", killServer);
