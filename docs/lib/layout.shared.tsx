import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";

export const gitConfig = {
  user: "anaslimem",
  repo: "cortexadb",
  branch: "main",
};

export const homeOptions = {
  nav: {
    title: "CortexaDB",
    icon: (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        className="w-6 h-6"
        stroke="currentColor"
        strokeWidth="2"
        aria-label="CortexaDB Logo"
      >
        <path d="M12 2L2 7l10 5 10-5-10-5z" />
        <path d="M2 17l10 5 10-5" />
        <path d="M2 12l10 5 10-5" />
      </svg>
    ),
  },
  links: [
    {
      text: "Documentation",
      url: "/docs",
    },
    {
      text: "GitHub",
      url: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
    },
  ],
};

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: "CortexaDB",
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
