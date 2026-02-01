// .markdownlint-custom-rules.js
export default [
    {
        names: ["blank-line-before-math"],
        description:
            "Blank line required before opening $$ or \\[, and after closing $$",
        tags: ["spacing"],
        function: (params, onError) => {
            const { lines } = params;
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                const trimmedLine = line.trim();

                // Check for opening $$ or \[ at the start of line
                const isOpeningMath =
                    (trimmedLine.startsWith("$$") &&
                        !trimmedLine.startsWith("$$$")) ||
                    trimmedLine.startsWith("\\[");

                // Check for closing $$ at the end of line
                const isClosingMath =
                    trimmedLine.endsWith("$$") && trimmedLine !== "$$";

                // Blank line before opening math
                if (isOpeningMath && i > 0 && lines[i - 1].trim() !== "") {
                    onError({
                        lineNumber: i + 1,
                        detail: `Blank line required before math delimiter`,
                    });
                }

                // Blank line after closing math
                if (
                    isClosingMath &&
                    i < lines.length - 1 &&
                    lines[i + 1].trim() !== ""
                ) {
                    onError({
                        lineNumber: i + 1,
                        detail: `Blank line required after closing math delimiter`,
                    });
                }
            }
        },
    },
];
