// .markdownlint-custom-rules.js
export default [
  {
    names: ["blank-line-before-math"],
    description: "Blank line required before $$ or \\[",
    tags: ["spacing"],
    function: (params, onError) => {
      const { lines } = params;
      for (let i = 1; i < lines.length; i++) {
        const line = lines[i];
        const prevLine = lines[i - 1];

        // Check if line starts with $$ or \[
        if ((line.trim().startsWith("$$") || line.trim().startsWith("\\[")) &&
            prevLine.trim() !== "") {
          onError({
            lineNumber: i + 1,
            detail: `Blank line required before math delimiter`,
          });
        }
      }
    },
  },
];
