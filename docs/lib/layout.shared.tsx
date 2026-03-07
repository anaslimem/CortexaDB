import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

export const gitConfig = {
  user: 'anaslimem',
  repo: 'cortexadb',
  branch: 'main',
};

export const homeOptions = {
  nav: {
    title: 'CortexaDB',
    icon: (
      <img src="/logo.png" className="w-8 h-8 rounded-md" alt="CortexaDB Logo" />
    ),
  },
  links: [
    {
      text: 'Documentation',
      url: '/docs',
    },
    {
      text: 'GitHub',
      url: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
    },
  ],
};

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: (
        <div className="flex items-center gap-2">
          <img src="/logo.png" className="w-6 h-6 rounded" alt="CortexaDB Logo" />
          <span className="font-semibold">CortexaDB</span>
        </div>
      ),
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
