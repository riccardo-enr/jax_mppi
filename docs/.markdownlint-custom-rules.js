// .markdownlint-custom-rules.js
export default [
    {
        names: ["blank-line-before-math"],
        description: "Blank line required before and after block-level $$ math",
        tags: ["spacing"],
        function: (params, onError) => {
            const { lines } = params;
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                const trimmedLine = line.trim();

                // Only check for block-level math (lines that are exactly $$)
                const isBlockMath = trimmedLine === "$$";

                // Blank line before block math
                if (isBlockMath && i > 0 && lines[i - 1].trim() !== "") {
                    onError({
                        lineNumber: i + 1,
                        detail: `Blank line required before block math`,
                    });
                }

                // Blank line after block math
                if (
                    isBlockMath &&
                    i < lines.length - 1 &&
                    lines[i + 1].trim() !== ""
                ) {
                    onError({
                        lineNumber: i + 1,
                        detail: `Blank line required after block math`,
                    });
                }
            }
        },
    },
];
