// electron-builder afterPack 钩子：把 PureCut 的 Python 引擎（scripts/）与离线检测模型（models/）
// 拷贝到安装包的 resources 目录（asar 之外——Python 无法从只读 asar 中读取 .py / .onnx / .tflite），
// 并校验内置 ffmpeg/ffprobe 可执行。
// 说明：Python 解释器与第三方依赖（fastapi / mediapipe / onnxruntime 等）不在此拷贝范围内，
// 运行机器需预装 Python 并 `pip install -r requirements.txt`，或用 PyInstaller 把引擎打成可执行文件后随包分发。
const path = require("path");
const fs = require("fs");
const { execSync } = require("child_process");
const { Arch } = require("builder-util");
const { prepareBundledMediaTools } = require("./media-tools.cjs");

/** 递归拷贝目录树（源是真实文件、无 symlink，任何平台拷贝均安全） */
function copyTree(src, dest) {
  fs.rmSync(dest, { recursive: true, force: true });
  fs.mkdirSync(dest, { recursive: true });
  if (process.platform === "win32") {
    execSync(`robocopy "${src}" "${dest}" /e /nfl /ndl /njh /njs >NUL || ver>NUL`, { shell: "cmd.exe" });
  } else {
    execSync(`cp -R "${src}/." "${dest}/"`);
  }
}

exports.default = async function afterPack(context) {
  const { appOutDir, packager, electronPlatformName } = context;
  const productName = packager.appInfo.productFilename;

  const resourcesDir =
    electronPlatformName === "darwin"
      ? path.join(appOutDir, `${productName}.app`, "Contents", "Resources")
      : path.join(appOutDir, "resources");

  const root = process.cwd();

  // 1) Python 引擎（scripts/）
  const scriptsSrc = path.join(root, "scripts");
  if (!fs.existsSync(scriptsSrc)) throw new Error("[afterPack] 未找到 scripts/，请确认 Python 引擎存在");
  copyTree(scriptsSrc, path.join(resourcesDir, "scripts"));

  // 2) 离线检测模型（models/）
  const modelsSrc = path.join(root, "models");
  if (!fs.existsSync(modelsSrc)) throw new Error("[afterPack] 未找到 models/，请确认离线检测模型存在");
  copyTree(modelsSrc, path.join(resourcesDir, "models"));

  console.log("[afterPack] Python 引擎与离线模型已拷入 resources ✓");

  // 3) ffmpeg/ffprobe 权限与运行自检
  prepareBundledMediaTools(resourcesDir, electronPlatformName, Arch[context.arch]);
  console.log("[afterPack] ffmpeg / ffprobe 权限与运行自检通过 ✓");
};
