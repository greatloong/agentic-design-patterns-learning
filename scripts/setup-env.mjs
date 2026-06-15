#!/usr/bin/env node
/**
 * 统一环境变量自助脚本（在 `pnpm install` 的 postinstall 阶段自动运行）。
 *
 * 做两件事：
 *   1. 若根目录缺少 .env，则从 .env.example 复制一份（并提示去填真实值）。
 *   2. 为 packages/* 下每个包创建/修复指向根 .env 的软链接（.env -> ../../.env）。
 *
 * 设计为幂等：重复运行不会报错、不会破坏已正确的链接。
 * clone 仓库后只需 `pnpm install` 即可自动完成，无需手动建软链接。
 */
import {
  readdirSync,
  existsSync,
  lstatSync,
  readlinkSync,
  symlinkSync,
  rmSync,
  copyFileSync,
} from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const rootEnv = join(repoRoot, ".env");
const rootEnvExample = join(repoRoot, ".env.example");
const packagesDir = join(repoRoot, "packages");

const LINK_TARGET = join("..", "..", ".env"); // packages/<pkg>/.env -> ../../.env

const isSymlink = (p) => {
  try {
    return lstatSync(p).isSymbolicLink();
  } catch {
    return false;
  }
};

// 1) 确保根 .env 存在
if (!existsSync(rootEnv)) {
  if (existsSync(rootEnvExample)) {
    copyFileSync(rootEnvExample, rootEnv);
    console.log("[setup-env] 已从 .env.example 创建根 .env —— 请填入真实的 API Key。");
  } else {
    console.warn(
      "[setup-env] 警告：根目录缺少 .env 且没有 .env.example，软链接将指向不存在的文件。"
    );
  }
}

// 2) 为每个包创建/修复软链接
if (!existsSync(packagesDir)) {
  console.warn("[setup-env] 未找到 packages/ 目录，跳过。");
  process.exit(0);
}

const pkgs = readdirSync(packagesDir, { withFileTypes: true })
  .filter((d) => d.isDirectory())
  .map((d) => d.name);

let created = 0;
let fixed = 0;
let ok = 0;

for (const pkg of pkgs) {
  const linkPath = join(packagesDir, pkg, ".env");
  const linkExists = isSymlink(linkPath) || existsSync(linkPath);

  if (isSymlink(linkPath) && readlinkSync(linkPath) === LINK_TARGET) {
    ok += 1;
    continue; // 已是正确的软链接
  }

  if (linkExists) {
    // 普通文件或指向别处的软链接：删除后重建，确保单一数据源
    rmSync(linkPath, { force: true });
    symlinkSync(LINK_TARGET, linkPath);
    fixed += 1;
  } else {
    symlinkSync(LINK_TARGET, linkPath);
    created += 1;
  }
}

console.log(
  `[setup-env] 完成：共 ${pkgs.length} 个包` +
    `（新建 ${created} / 修复 ${fixed} / 已正确 ${ok}）`
);
