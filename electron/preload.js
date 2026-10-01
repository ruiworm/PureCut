const { contextBridge, ipcRenderer, webUtils } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Window controls
  minimize: () => ipcRenderer.send('window-minimize'),
  maximize: () => ipcRenderer.send('window-maximize'),
  close: () => ipcRenderer.send('window-close'),
  isMaximized: () => ipcRenderer.invoke('window-is-maximized'),

  // Native File Dialogs
  selectDirectory: (defaultPath) => ipcRenderer.invoke('dialog-select-directory', defaultPath),
  selectVideoFiles: () => ipcRenderer.invoke('dialog-select-videos'),

  // Native Path extraction from Drag-and-Drop File object
  getPathForFile: (file) => {
    try {
      if (webUtils && typeof webUtils.getPathForFile === 'function') {
        return webUtils.getPathForFile(file);
      }
      return file.path || '';
    } catch {
      return file.path || '';
    }
  },

  // Shell integration
  showItemInFolder: (fullPath) => ipcRenderer.send('shell-show-item', fullPath),
  openPath: (fullPath) => ipcRenderer.invoke('shell-open-path', fullPath)
});

