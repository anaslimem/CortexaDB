'use client';

const code = `from cortexadb import CortexaDB
from cortexadb.providers.openai import OpenAIEmbedder

db = CortexaDB.open("agent.mem", embedder=OpenAIEmbedder())

# Store memories
db.add("User prefers dark mode")
db.add("User works at Stripe")

# Semantic search
hits = db.query("What does the user like?").execute()
# => [Hit(id=1, score=0.87), Hit(id=2, score=0.72)]`;

const kw = 'color: #D73A49'; // keyword
const imp = 'color: #6F42C1'; // import/module
const str = 'color: #032F62'; // string
const fn = 'color: #6F42C1'; // function
const cm = 'color: #6A737D'; // comment

const lines = [
  { style: kw, text: 'from' },
  { style: imp, text: 'cortexadb' },
  { style: kw, text: 'import' },
  { style: imp, text: 'CortexaDB' },
  { style: '', text: '' },
  { style: kw, text: 'from' },
  { style: '', text: 'cortexadb.providers.openai' },
  { style: kw, text: 'import' },
  { style: imp, text: 'OpenAIEmbedder' },
  { style: '', text: '' },
  { style: '', text: 'db' },
  { style: '', text: '=' },
  { style: imp, text: 'CortexaDB' },
  { style: fn, text: '.open' },
  { style: '', text: '("agent.mem", embedder=' },
  { style: imp, text: 'OpenAIEmbedder' },
  { style: '', text: '())' },
  { style: '', text: '' },
  { style: cm, text: '# Store memories' },
  { style: '', text: 'db.add("User prefers dark mode")' },
  { style: '', text: 'db.add("User works at Stripe")' },
  { style: '', text: '' },
  { style: cm, text: '# Semantic search' },
  { style: '', text: 'hits = db.query("What does the user like?").execute()' },
  { style: cm, text: '# => [Hit(id=1, score=0.87), Hit(id=2, score=0.72)]' },
];

export function CodePreview() {
  return (
    <figure className="my-4 bg-fd-card rounded-xl shiki relative border shadow-sm not-prose overflow-hidden text-sm">
      <div className="flex items-center gap-2 px-4 py-3 border-b border-fd-border/50">
        <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
        <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
        <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
        <span className="ml-2 text-sm text-fd-muted-foreground">agent.py</span>
      </div>
      <pre className="p-6 text-sm overflow-x-auto font-mono leading-relaxed">
        <code className="text-fd-foreground">
          {lines.map((line, idx) => (
            <span key={`${line.text}-${idx}`} style={{ color: line.style || 'inherit' }}>
              {line.text || '\u00A0'}
              {idx < lines.length - 1 && '\n'}
            </span>
          ))}
        </code>
      </pre>
    </figure>
  );
}
