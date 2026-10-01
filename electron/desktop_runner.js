const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

let mainWindow = null;
let pythonProcess = null;
const PORT = process.env.PORT || 8765;
const APP_URL = `http://127.0.0.1:${PORT}`;

const PING_URL = `http://127.0.0.1:${PORT}/api/settings/directories`;

/** Ping if local Python API service is running */
function checkServerRunning() {
  return new Promise((resolve) => {
    const req = http.get(PING_URL, (res) => {
      resolve(Boolean(res.statusCode && res.statusCode < 500));
    });
    req.on('error', () => resolve(false));
    req.setTimeout(2500, () => {
      req.destroy();
      resolve(false);
    });
  });
}

/** Check if local Python API service is ready with polling */
function waitForServer(timeoutMs = 25000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const check = () => {
      const req = http.get(PING_URL, (res) => {
        if (res.statusCode && res.statusCode < 500) {
          resolve(true);
        } else {
          retry();
        }
      });
      req.on('error', () => retry());
      req.end();
    };

    const retry = () => {
      if (Date.now() - start > timeoutMs) {
        reject(new Error('Python 算法服务启动超时'));
      } else {
        setTimeout(check, 300);
      }
    };

    check();
  });
}

/** Spawn Python sidecar background daemon */
function startPythonSubprocess() {
  const scriptPath = path.join(__dirname, '..', 'scripts', 'web_studio.py');
  console.log(`[Electron] 正在启动 Python 核心算法引擎: python ${scriptPath} ${PORT}`);

  pythonProcess = spawn('python', [scriptPath, String(PORT)], {
    cwd: path.join(__dirname, '..'),
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    stdio: 'ignore'
  });

  pythonProcess.on('error', (err) => {
    console.error('[Electron] Python 进程启动失败:', err);
  });

  pythonProcess.on('exit', (code, signal) => {
    console.log(`[Electron] Python 进程已退出 (code: ${code}, signal: ${signal})`);
  });
}

/** Kill Python child process gracefully */
function killPythonSubprocess() {
  if (pythonProcess) {
    try {
      console.log('[Electron] 正在安全清理 Python 子进程...');
      pythonProcess.kill();
    } catch {
      // ignore
    }
    pythonProcess = null;
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1100,
    minHeight: 720,
    frame: false, // Frameless window for premium NLE dark desktop UI
    backgroundColor: '#0c0d10',
    title: 'PureCut Studio Pro · 智能矩阵剪辑工作台',
    icon: path.join(__dirname, 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false // Allow loading local file protocols & streaming
    }
  });

  mainWindow.loadURL(APP_URL);

  // Dev Hotkey: F12 for DevTools, F5 / Ctrl+R for reload
  mainWindow.webContents.on('before-input-event', (event, input) => {
    if (input.key === 'F12') {
      mainWindow.webContents.toggleDevTools();
      event.preventDefault();
    } else if (input.key === 'F5' || (input.control && input.key.toLowerCase() === 'r')) {
      mainWindow.webContents.reload();
      event.preventDefault();
    }
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// IPC Window Controls
ipcMain.on('window-minimize', () => {
  if (mainWindow) mainWindow.minimize();
});

ipcMain.on('window-maximize', () => {
  if (mainWindow) {
    if (mainWindow.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow.maximize();
    }
  }
});

ipcMain.handle('window-is-maximized', () => {
  return mainWindow ? mainWindow.isMaximized() : false;
});

ipcMain.on('window-close', () => {
  if (mainWindow) mainWindow.close();
});

// IPC Native Dialogs
ipcMain.handle('dialog-select-directory', async (event, defaultPath) => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    title: '选择包含视频素材的文件夹',
    defaultPath: defaultPath || undefined,
    properties: ['openDirectory', 'dontAddToRecent']
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

ipcMain.handle('dialog-select-videos', async () => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    title: '选择视频素材文件',
    filters: [
      { name: '视频文件', extensions: ['mp4', 'mov', 'webm', 'mkv', 'avi', 'm4v', 'flv', 'wmv'] },
      { name: '所有文件', extensions: ['*'] }
    ],
    properties: ['openFile', 'multiSelections', 'dontAddToRecent']
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths;
});

ipcMain.on('shell-show-item', (event, fullPath) => {
  if (fullPath) shell.showItemInFolder(fullPath);
});

ipcMain.handle('shell-open-path', async (event, fullPath) => {
  if (fullPath) return await shell.openPath(fullPath);
  return '';
});

// Lifecycle
app.whenReady().then(async () => {
  try {
    const isRunning = await checkServerRunning();
    if (!isRunning) {
      startPythonSubprocess();
      await waitForServer();
    } else {
      console.log(`[Electron] 检测到已有核心引擎运行在端口 ${PORT}，直接连接！`);
    }
    createWindow();
  } catch (err) {
    dialog.showErrorBox('PureCut 启动失败', err.message);
    app.quit();
  }
});

app.on('window-all-closed', () => {
  killPythonSubprocess();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  killPythonSubprocess();
});
