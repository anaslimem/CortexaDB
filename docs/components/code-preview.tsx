"use client";

const kw = "color: #D73A49"; // keyword
const imp = "color: #6F42C1"; // import/module
const fn = "color: #6F42C1"; // function
const cm = "color: #6A737D"; // comment

const lines: { text: string; style?: string; id: string }[] = [
  { id: "1", style: kw, text: "from" },
  { id: "2", style: imp, text: "cortexadb" },
  { id: "3", style: kw, text: "import" },
  { id: "4", style: imp, text: "CortexaDB" },
  { id: "5", text: "" },
  { id: "6", style: kw, text: "from" },
  { id: "7", text: "cortexadb.providers.openai" },
  { id: "8", style: kw, text: "import" },
  { id: "9", style: imp, text: "OpenAIEmbedder" },
  { id: "10", text: "" },
  { id: "11", text: "db" },
  { id: "12", text: "= " },
  { id: "13", style: imp, text: "CortexaDB" },
  { id: "14", style: fn, text: ".open" },
  { id: "15", text: '("agent.mem", embedder=' },
  { id: "16", style: imp, text: "OpenAIEmbedder" },
  { id: "17", text: "())" },
  { id: "18", text: "" },
  { id: "19", style: cm, text: "# Store memories" },
  { id: "20", text: 'db.remember("User prefers dark mode")' },
  { id: "21", text: 'db.remember("User works at Stripe")' },
  { id: "22", text: "" },
  { id: "23", style: cm, text: "# Semantic search" },
  { id: "24", text: 'hits = db.ask("What does the user like?")' },
  {
    id: "25",
    style: cm,
    text: "# => [Hit(id=1, score=0.87), Hit(id=2, score=0.72)]",
  },
];

export function CodePreview() {
  return (
    <figure className="my-4 bg-[#24292e] rounded-xl shiki relative border shadow-sm not-prose overflow-hidden text-sm dark:bg-[#24292e]">
      <div className="flex items-center gap-2 px-4 py-3 border-b border-white/10">
        <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
        <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
        <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
        <span className="ml-2 text-sm text-white/60">agent.py</span>
      </div>
      <pre className="p-6 text-sm overflow-x-auto font-mono leading-relaxed text-[#e1e4e8]">
        <code>
          {lines.map((line) => (
            <span
              key={line.id}
              style={{ color: line.style }}
              className={line.style}
            >
              {line.text || "\u00A0"}
            </span>
          ))}
        </code>
      </pre>
    </figure>
  );
}
