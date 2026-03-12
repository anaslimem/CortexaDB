import Link from 'next/link';
import { ArrowRight, Database, GitBranch, Zap, Shield, Layers, Github, Star, Download, TrendingUp } from 'lucide-react';
import { CodePreview } from '@/components/code-preview';

const features = [
  {
    icon: Database,
    title: 'Hybrid Retrieval',
    description: 'Combine vector similarity, graph relations, and recency in a single query',
  },
  {
    icon: Layers,
    title: 'Smart Chunking',
    description: '5 strategies for document ingestion: fixed, recursive, semantic, markdown, json',
  },
  {
    icon: Zap,
    title: 'HNSW Indexing',
    description: 'Ultra-fast approximate nearest neighbor search via USearch',
  },
  {
    icon: GitBranch,
    title: 'Knowledge Graphs',
    description: 'Connect memories with directed edges and traverse them with BFS',
  },
  {
    icon: Shield,
    title: 'Hard Durability',
    description: 'WAL and segmented storage ensure crash safety and data integrity',
  },
  {
    icon: Database,
    title: 'Multi-Agent Collections',
    description: 'Isolate memories between agents within a single database file',
  },
];

export default function HomePage() {
  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)]">
      {/* Hero Section */}
      <section className="flex-1 flex flex-col items-center justify-center py-24 px-4 text-center">
        <div className="max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-fd-primary/10 text-fd-primary text-sm font-medium mb-8">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-fd-primary opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-fd-primary"></span>
            </span>
            Now in Beta
          </div>

          <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6 bg-linear-to-b from-fd-foreground to-fd-foreground/70 bg-clip-text text-transparent">
            The database for
            <br />
            AI Agent Memory
          </h1>

          <p className="text-xl text-fd-muted-foreground max-w-2xl mx-auto mb-10">
            CortexaDB is a simple, fast, and hard-durable embedded database designed specifically for AI agent memory. Single-file, no server required.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/docs/getting-started/installation"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-fd-primary text-fd-primary-foreground font-medium hover:opacity-90 transition-opacity"
            >
              Get Started
              <ArrowRight className="w-4 h-4" />
            </Link>
            <a
              href="https://github.com/anaslimem/CortexaDB"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg border border-fd-border bg-fd-card text-fd-foreground font-medium hover:bg-fd-accent transition-colors"
            >
              <Github className="w-4 h-4" />
              Star on GitHub
              <Star className="w-4 h-4 fill-yellow-500 text-yellow-500" />
            </a>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-12 border-y border-fd-border bg-fd-muted/10">
        <div className="max-w-6xl mx-auto px-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            <div className="flex flex-col items-center text-center space-y-2">
              <div className="p-3 rounded-xl bg-yellow-500/10 text-yellow-600 dark:text-yellow-500">
                <Star className="w-6 h-6 fill-current" />
              </div>
              <div className="text-3xl font-bold tracking-tight">27</div>
              <div className="text-sm text-fd-muted-foreground font-medium uppercase tracking-wider">GitHub Stars</div>
            </div>
            <div className="flex flex-col items-center text-center space-y-2">
              <div className="p-3 rounded-xl bg-blue-500/10 text-blue-600 dark:text-blue-500">
                <Download className="w-6 h-6 transition-transform group-hover:scale-110" />
              </div>
              <div className="text-3xl font-bold tracking-tight">4.7k+</div>
              <div className="text-sm text-fd-muted-foreground font-medium uppercase tracking-wider">PyPI Downloads</div>
            </div>
            <div className="flex flex-col items-center text-center space-y-2">
              <div className="p-3 rounded-xl bg-purple-500/10 text-purple-600 dark:text-purple-500">
                <Zap className="w-6 h-6 fill-current" />
              </div>
              <div className="text-3xl font-bold tracking-tight text-fd-primary">103x</div>
              <div className="text-sm text-fd-muted-foreground font-medium uppercase tracking-wider">Faster Ingestion</div>
            </div>
            <div className="flex flex-col items-center text-center space-y-2">
              <div className="p-3 rounded-xl bg-green-500/10 text-green-600 dark:text-green-500">
                <Shield className="w-6 h-6" />
              </div>
              <div className="text-3xl font-bold tracking-tight">Beta</div>
              <div className="text-sm text-fd-muted-foreground font-medium uppercase tracking-wider">Production Ready</div>
            </div>
          </div>
        </div>
      </section>

      {/* Code Preview */}
      <section className="py-16 px-4">
        <div className="max-w-3xl mx-auto">
          <div className="rounded-xl border border-fd-border bg-fd-card overflow-hidden shadow-xl">
            <div className="flex items-center gap-2 px-4 py-3 border-b border-fd-border bg-fd-muted/50">
              <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
              <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
              <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
              <span className="ml-2 text-sm text-fd-muted-foreground">agent.py</span>
            </div>
            <pre className="p-6 text-sm overflow-x-auto">
              <code className="text-fd-foreground">
                {`from cortexadb import CortexaDB
from cortexadb.providers.openai import OpenAIEmbedder

db = CortexaDB.open("agent.mem", embedder=OpenAIEmbedder())

# Store memories
db.add("User prefers dark mode")
db.add("User works at Stripe")

# Semantic search
hits = db.query("What does the user like?").execute()
# => [Hit(id=1, score=0.87), Hit(id=2, score=0.72)]`}
              </code>
            </pre>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-24 px-4 bg-fd-muted/30">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-4">
            Everything you need for agent memory
          </h2>
          <p className="text-fd-muted-foreground text-center mb-16 max-w-2xl mx-auto">
            Built from the ground up for AI agents with hybrid retrieval, knowledge graphs, and rock-solid durability.
          </p>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature) => (
              <div
                key={feature.title}
                className="p-6 rounded-xl border border-fd-border bg-fd-card hover:border-fd-primary/50 transition-colors group"
              >
                <div className="w-12 h-12 rounded-lg bg-fd-primary/10 flex items-center justify-center mb-4 group-hover:bg-fd-primary/20 transition-colors">
                  <feature.icon className="w-6 h-6 text-fd-primary" />
                </div>
                <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                <p className="text-sm text-fd-muted-foreground">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Quick Links */}
      <section className="py-24 px-4">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">Explore the docs</h2>
          <div className="grid md:grid-cols-3 gap-6">
            <Link
              href="/docs/getting-started/quickstart"
              className="p-6 rounded-xl border border-fd-border bg-fd-card hover:border-fd-primary/50 transition-colors"
            >
              <h3 className="text-lg font-semibold mb-2">Quickstart</h3>
              <p className="text-sm text-fd-muted-foreground">Get up and running in 5 minutes</p>
            </Link>
            <Link
              href="/docs/guides/core-concepts"
              className="p-6 rounded-xl border border-fd-border bg-fd-card hover:border-fd-primary/50 transition-colors"
            >
              <h3 className="text-lg font-semibold mb-2">Architecture</h3>
              <p className="text-sm text-fd-muted-foreground">Understand how CortexaDB works</p>
            </Link>
            <Link
              href="/docs/api/python"
              className="p-6 rounded-xl border border-fd-border bg-fd-card hover:border-fd-primary/50 transition-colors"
            >
              <h3 className="text-lg font-semibold mb-2">API Reference</h3>
              <p className="text-sm text-fd-muted-foreground">Complete Python & Rust APIs</p>
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 border-t border-fd-border">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-sm text-fd-muted-foreground">
            Released under MIT and Apache-2.0 licenses
          </p>
          <div className="flex items-center gap-6 text-sm text-fd-muted-foreground">
            <a
              href="https://github.com/anaslimem/cortexadb"
              className="hover:text-fd-foreground transition-colors"
            >
              GitHub
            </a>
            <a
              href="/docs"
              className="hover:text-fd-foreground transition-colors"
            >
              Documentation
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
