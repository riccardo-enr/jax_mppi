#!/usr/bin/env node
// Auto-fix script for blank-line-before-math rule

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

function getAllMarkdownFiles(dir) {
    const files = [];
    const items = fs.readdirSync(dir);

    items.forEach((item) => {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);

        if (stat.isDirectory()) {
            if (!["node_modules", ".venv"].includes(item)) {
                files.push(...getAllMarkdownFiles(fullPath));
            }
        } else if (item.endsWith(".md")) {
            files.push(fullPath);
        }
    });

    return files;
}

// Get all markdown files
const files = getAllMarkdownFiles(__dirname);
let fixedCount = 0;

files.forEach((filePath) => {
    let content = fs.readFileSync(filePath, "utf-8");
    const lines = content.split("\n");
    let modified = false;

    for (let i = lines.length - 1; i >= 1; i--) {
        const line = lines[i];
        const prevLine = lines[i - 1];

        // Check if line starts with $$ or \[
        if (
            (line.trim().startsWith("$$") || line.trim().startsWith("\\[")) &&
            prevLine.trim() !== ""
        ) {
            // Insert blank line before
            lines.splice(i, 0, "");
            modified = true;
            console.log(
                `✓ Fixed: ${path.relative(process.cwd(), filePath)}:${i + 1}`,
            );
            fixedCount++;
        }
    }

    if (modified) {
        fs.writeFileSync(filePath, lines.join("\n"), "utf-8");
    }
});

console.log(`✅ Auto-fix complete! Fixed ${fixedCount} violations.`);
