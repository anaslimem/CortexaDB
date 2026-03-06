import {
  ArrowRight,
  Database,
  GitBranch,
  GitFork,
  Github,
  Layers,
  Shield,
  Star,
  Zap,
} from "lucide-react";
import Link from "next/link";
import { getGitHubStats } from "@/lib/github";

const features = [
  {
    icon: Database,
    title: "Hybrid Retrieval",
    description:
      "Combine vector similarity, graph relations, and recency in a single query",
  },
  {
    icon: Layers,
    title: "Smart Chunking",
    description:
      "5 strategies for document ingestion: fixed, recursive, semantic, markdown, json",
  },
  {
    icon: Zap,
    title: "HNSW Indexing",
    description: "Ultra-fast approximate nearest neighbor search via USearch",
  },
  {
    icon: GitBranch,
    title: "Knowledge Graphs",
    description:
      "Connect memories with directed edges and traverse them with BFS",
  },
  {
    icon: Shield,
    title: "Hard Durability",
    description:
      "WAL and segmented storage ensure crash safety and data integrity",
  },
  {
    icon: Database,
    title: "Multi-Agent Namespaces",
    description:
      "Isolate memories between agents within a single database file",
  },
];

export default async function HomePage() {
  const githubStats = await getGitHubStats();

  return (
    <div className="flex flex-col min-h-[calc(100vh-4rem)]">
      {/* Hero Section */}
      <section className="flex-1 flex flex-col items-center justify-center py-24 px-4 text-center relative overflow-hidden">
        {/* Background decoration */}
        <div className="absolute inset-0 -z-10 overflow-hidden">
          <div
            className="absolute top-[10%] left-[20%] w-125 h-125 bg-fd-primary/10 rounded-full blur-3xl animate-pulse"
            style={{ animationDuration: "4s" }}
          />
          <div
            className="absolute bottom-[10%] right-[20%] w-100 h-100 bg-fd-primary/8 rounded-full blur-3xl animate-pulse"
            style={{ animationDuration: "6s", animationDelay: "1s" }}
          />
          <div
            className="absolute top-[50%] left-[50%] w-75 h-75 bg-fd-primary/5 rounded-full blur-3xl animate-pulse"
            style={{ animationDuration: "5s", animationDelay: "2s" }}
          />

          {/* Animated nodes/dots representing vector embeddings */}
          <div className="absolute inset-0" aria-hidden="true">
            <svg
              role="img"
              aria-label="Vector nodes animation"
              className="w-full h-full opacity-30"
              preserveAspectRatio="xMidYMid slice"
            >
              <defs>
                <radialGradient id="nodeGradient" cx="50%" cy="50%" r="50%">
                  <stop
                    offset="0%"
                    stopColor="currentColor"
                    stopOpacity="0.8"
                  />
                  <stop
                    offset="100%"
                    stopColor="currentColor"
                    stopOpacity="0"
                  />
                </radialGradient>
                <linearGradient
                  id="lineGradient"
                  x1="0%"
                  y1="0%"
                  x2="100%"
                  y2="0%"
                >
                  <stop offset="0%" stopColor="currentColor" stopOpacity="0" />
                  <stop
                    offset="50%"
                    stopColor="currentColor"
                    stopOpacity="0.6"
                  />
                  <stop
                    offset="100%"
                    stopColor="currentColor"
                    stopOpacity="0"
                  />
                </linearGradient>
                <style>{`
                  @keyframes float {
                    0%, 100% { transform: translateY(0px); }
                    50% { transform: translateY(-8px); }
                  }
                  @keyframes dataFlow {
                    0% { stroke-dashoffset: 20; }
                    100% { stroke-dashoffset: 0; }
                  }
                  @keyframes glow {
                    0%, 100% { opacity: 0.6; }
                    50% { opacity: 1; }
                  }
                  .node-float { animation: float 4s ease-in-out infinite; }
                  .node-float-delay-1 { animation-delay: 0.5s; }
                  .node-float-delay-2 { animation-delay: 1s; }
                  .node-float-delay-3 { animation-delay: 1.5s; }
                  .node-float-delay-4 { animation-delay: 2s; }
                  .line-flow { stroke-dasharray: 5,5; animation: dataFlow 2s linear infinite; }
                  .line-flow-delay { animation-delay: 1s; }
                  .node-glow { animation: glow 3s ease-in-out infinite; }
                `}</style>
              </defs>
              {/* Connection lines */}
              <g
                stroke="url(#lineGradient)"
                strokeWidth="1"
                className="text-fd-primary/40"
              >
                <line
                  x1="15%"
                  y1="20%"
                  x2="25%"
                  y2="40%"
                  className="line-flow"
                />
                <line
                  x1="25%"
                  y1="40%"
                  x2="45%"
                  y2="35%"
                  className="line-flow line-flow-delay"
                />
                <line
                  x1="45%"
                  y1="35%"
                  x2="60%"
                  y2="50%"
                  className="line-flow"
                />
                <line
                  x1="60%"
                  y1="50%"
                  x2="80%"
                  y2="30%"
                  className="line-flow line-flow-delay"
                />
                <line
                  x1="80%"
                  y1="30%"
                  x2="85%"
                  y2="60%"
                  className="line-flow"
                />
                <line
                  x1="15%"
                  y1="20%"
                  x2="10%"
                  y2="50%"
                  className="line-flow line-flow-delay"
                />
                <line
                  x1="10%"
                  y1="50%"
                  x2="25%"
                  y2="70%"
                  className="line-flow"
                />
                <line
                  x1="25%"
                  y1="70%"
                  x2="50%"
                  y2="75%"
                  className="line-flow line-flow-delay"
                />
                <line
                  x1="50%"
                  y1="75%"
                  x2="70%"
                  y2="65%"
                  className="line-flow"
                />
                <line
                  x1="70%"
                  y1="65%"
                  x2="85%"
                  y2="60%"
                  className="line-flow line-flow-delay"
                />
                <line
                  x1="45%"
                  y1="35%"
                  x2="50%"
                  y2="55%"
                  className="line-flow"
                />
                <line
                  x1="50%"
                  y1="55%"
                  x2="70%"
                  y2="65%"
                  className="line-flow line-flow-delay"
                />
              </g>
              {/* Nodes */}
              <g fill="currentColor" className="text-fd-primary">
                <circle
                  cx="15%"
                  cy="20%"
                  r="4"
                  className="node-float node-glow"
                />
                <circle
                  cx="25%"
                  cy="40%"
                  r="3"
                  className="node-float node-float-delay-1 node-glow"
                />
                <circle
                  cx="45%"
                  cy="35%"
                  r="5"
                  className="node-float node-float-delay-2 node-glow"
                />
                <circle
                  cx="60%"
                  cy="50%"
                  r="3"
                  className="node-float node-float-delay-3 node-glow"
                />
                <circle
                  cx="80%"
                  cy="30%"
                  r="4"
                  className="node-float node-float-delay-4 node-glow"
                />
                <circle
                  cx="85%"
                  cy="60%"
                  r="3"
                  className="node-float node-glow"
                />
                <circle
                  cx="10%"
                  cy="50%"
                  r="3"
                  className="node-float node-float-delay-1 node-glow"
                />
                <circle
                  cx="25%"
                  cy="70%"
                  r="4"
                  className="node-float node-float-delay-2 node-glow"
                />
                <circle
                  cx="50%"
                  cy="75%"
                  r="3"
                  className="node-float node-float-delay-3 node-glow"
                />
                <circle
                  cx="70%"
                  cy="65%"
                  r="5"
                  className="node-float node-float-delay-4 node-glow"
                />
                <circle
                  cx="50%"
                  cy="55%"
                  r="3"
                  className="node-float node-glow"
                />
              </g>
              {/* Data particles traveling along lines */}
              <g fill="currentColor" className="text-fd-primary">
                <circle r="2" className="node-glow">
                  <animateMotion
                    dur="3s"
                    repeatCount="indefinite"
                    path="M15,20 L25,40 L45,35 L60,50 L80,30 L85,60"
                  />
                </circle>
                <circle r="2" className="node-glow">
                  <animateMotion
                    dur="4s"
                    repeatCount="indefinite"
                    path="M15,20 L10,50 L25,70 L50,75 L70,65 L85,60"
                  />
                </circle>
              </g>
            </svg>
          </div>

          {/* Grid pattern */}
          <div className="absolute inset-0 bg-[linear-gradient(rgba(128,128,128,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(128,128,128,0.03)_1px,transparent_1px)] bg-size-[50px_50px]" />
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-fd-primary/10 text-fd-primary text-sm font-medium mb-8 border border-fd-primary/20">
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
            Simple, fast, and hard-durable embedded database designed
            specifically for AI agents. Single file, no server, native vector +
            graph support.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-8">
            <Link
              href="/docs/getting-started/installation"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-fd-primary text-fd-primary-foreground font-medium hover:opacity-90 transition-all hover:scale-105"
            >
              <span>Get Started</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              href="/docs/guides/core-concepts"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg border border-fd-border text-fd-foreground font-medium hover:bg-fd-accent transition-colors"
            >
              Documentation
            </Link>
          </div>

          {/* GitHub CTA */}
          <div className="flex items-center justify-center gap-4">
            <a
              href="https://github.com/anaslimem/cortexadb"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-fd-border bg-fd-card/50 text-fd-muted-foreground hover:text-fd-foreground hover:border-fd-border/80 transition-colors"
            >
              <Github className="w-5 h-5" />
              <span className="text-sm font-medium">Star on GitHub</span>
            </a>
            <div className="flex items-center gap-1 text-sm text-fd-muted-foreground">
              <Star className="w-4 h-4 text-yellow-500" />
              <span>{githubStats.stars.toLocaleString()}</span>
            </div>
            <div className="flex items-center gap-1 text-sm text-fd-muted-foreground">
              <GitFork className="w-4 h-4" />
              <span>{githubStats.forks.toLocaleString()}</span>
            </div>
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
            Built from the ground up for AI agents with hybrid retrieval,
            knowledge graphs, and rock-solid durability.
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
                <p className="text-sm text-fd-muted-foreground">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Quick Links */}
      <section className="py-24 px-4">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">
            Explore the docs
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            <Link
              href="/docs/getting-started/quickstart"
              className="p-6 rounded-xl border border-fd-border bg-fd-card hover:border-fd-primary/50 transition-colors"
            >
              <h3 className="text-lg font-semibold mb-2">Quickstart</h3>
              <p className="text-sm text-fd-muted-foreground">
                Get up and running in 5 minutes
              </p>
            </Link>
            <Link
              href="/docs/guides/core-concepts"
              className="p-6 rounded-xl border border-fd-border bg-fd-card hover:border-fd-primary/50 transition-colors"
            >
              <h3 className="text-lg font-semibold mb-2">Architecture</h3>
              <p className="text-sm text-fd-muted-foreground">
                Understand how CortexaDB works
              </p>
            </Link>
            <Link
              href="/docs/api/python"
              className="p-6 rounded-xl border border-fd-border bg-fd-card hover:border-fd-primary/50 transition-colors"
            >
              <h3 className="text-lg font-semibold mb-2">API Reference</h3>
              <p className="text-sm text-fd-muted-foreground">
                Complete Python & Rust APIs
              </p>
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
