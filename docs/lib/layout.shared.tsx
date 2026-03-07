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
      title: 'CortexaDB',
      icon: <img src="/logo.png" className="w-6 h-6 rounded" alt="CortexaDB Logo" />,
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
